from typing import List, Dict, Tuple
import time
import tqdm
import uuid
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search.lexical.elastic_search import ElasticSearch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from .bing import search_bing_batch
import logging

logger = logging.getLogger(__name__)

def get_random_doc_id():
    return f'_{uuid.uuid4()}'


class SearchEngineConnector:
    def __init__(
        self,
        engine: str,
        only_domain: str = None,
        exclude_domains: List[str] = [],
    ):
        assert engine in {'bing'}
        self.engine = engine
        self.only_domain = only_domain
        self.exclude_domains = exclude_domains

    def retrieve(
        self,
        corpus = None,
        queries: Dict[int, str] = None,
        **kwargs,
    ):
        qs = list(queries.values())
        if self.engine == 'bing':
            all_results = search_bing_batch(
                qs, only_domain=self.only_domain, exclude_domains=self.exclude_domains)
        else:
            raise NotImplementedError
        qid2results: Dict[int, Dict[str, Tuple[float, str]]] = {}
        for (qid, query), results in zip(queries.items(), all_results):
            qid2results[qid] = {str(r['url']) + get_random_doc_id(): (0, r['snippet']) for r in results}
        return qid2results


class BM25:
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        index_name: str = None,
        engine: str = 'elasticsearch',
        **search_engine_kwargs,
    ):
        self.tokenizer = tokenizer
        # load index
        assert engine in {'elasticsearch', 'bing'}
        if engine == 'elasticsearch':
            self.max_ret_topk = 1000
            self.retriever = EvaluateRetrieval(
                BM25Search(index_name=index_name, hostname='localhost', initialize=False, number_of_shards=1),
                k_values=[self.max_ret_topk])
            self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 根据需求选择其他模型
        else:
            self.max_ret_topk = 50
            self.retriever = SearchEngineConnector(engine, **search_engine_kwargs)

    def retrieve(
        self,
        queries: List[str],  # (bs,)
        filter_ids: List[str] = None,  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
    ):
        assert topk <= self.max_ret_topk
        device = None
        bs = len(queries)

        # truncate queries
        if max_query_length:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            # truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # retrieve
        filter_ids = filter_ids or ([None] * len(queries))
        results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), list(zip(queries, filter_ids)))), disable_tqdm=True)

        # prepare outputs
        docids: List[str] = []
        docs: List[str] = []
        for qid, query in enumerate(queries):
            _docids: List[str] = []
            _docs: List[str] = []
            if qid in results:
                for did, (score, text) in results[qid].items():
                    _docids.append(did)
                    _docs.append(text)
                    if len(_docids) >= topk:
                        break
            if len(_docids) < topk:  # add dummy docs
                _docids += [get_random_doc_id() for _ in range(topk - len(_docids))]
                _docs += [''] * (topk - len(_docs))
            docids.extend(_docids)
            docs.extend(_docs)

        docids = np.array(docids).reshape(bs, topk)  # (bs, topk)
        docs = np.array(docs).reshape(bs, topk)  # (bs, topk)
        return docids, docs
    
    def k_means_retrieve(
        self,
        queries: List[str],  # (bs,)
        filter_ids: List[str] = None,  # (bs,)
        topk: int = 5,  # 每个查询检索的文档数量
        k: int = 2,  # 聚类的类别数量
        n: int = 2,  # 用于检索的倍数，最终检索 topk * n 个文档
        max_query_length: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于 K-Means 聚类的检索方法。

        参数：
            queries (List[str]): 查询列表。
            filter_ids (List[str], optional): 过滤的文档 ID 列表。
            topk (int): 每个查询检索的文档数量。
            k (int): 聚类的类别数量。
            n (int): 用于检索的倍数，最终检索 topk * n 个文档。
            max_query_length (int, optional): 查询的最大长度。

        返回：
            Tuple[np.ndarray, np.ndarray]: 返回文档 IDs 和文档内容的数组，形状为 (bs, k)。
        """
        assert topk * n <= self.max_ret_topk, f"topk * n ({topk * n}) 应小于等于 max_ret_topk ({self.max_ret_topk})"
        bs = len(queries)

        # 1. 截断查询（如果需要）
        if max_query_length:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            # truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # 2. 检索文档
        filter_ids = filter_ids or ([None] * len(queries))
        results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), list(zip(queries, filter_ids)))))

        # 3. 处理检索到的文档
        retrieved_points = []  # 收集文档点（例如，文档ID、内容等）
        for qid, query in enumerate(queries):
            if qid in results:
                for did, (score, text) in results[qid].items():
                    # print(f"retrieved_text : {text}")
                    retrieved_points.append({'id': did, 'content': text})

        # 4. 检查检索到的文档数量
        required_docs = topk * n
        if len(retrieved_points) < required_docs:
            logger.warning(f"检索到的文档数量 ({len(retrieved_points)}) 少于 topk * n ({required_docs})")
            required_docs = len(retrieved_points)

        # 5. 向量化文档内容
        documents = [point['content'] for point in retrieved_points[:required_docs]]
        document_vectors = self.embedding_model.encode(documents)

        # 6. 使用 K-Means 进行聚类
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X=document_vectors)

        # Unique clusters
        unique_clusters: set[int] = set(clusters)

        # Create a dictionary with the members of each cluster
        cluster_dict: defaultdict[int, list[int | None]] = defaultdict(list)
        for index, cluster in enumerate(clusters):
            cluster_dict[cluster].append(index)

        # M subsets
        m: int = min(len(indices) for indices in cluster_dict.values())
        m = min(m, topk)
        print(f"{m} document subsets will be created.")

        # Generate m unique subsets without replacement
        np.random.seed(seed=42)
        subsets: list[list[str]] = []
        
        # subsets的大小是m，也就是最小簇的大小
        # subsets的每个元素是一个长为k的列表，包含1~k簇中的某篇文档内容
        for _ in range(m):
            subset: list[int] = []
            for cluster in unique_clusters:
                chosen_element: int = np.random.choice(cluster_dict[cluster])
                subset.append(chosen_element)
                cluster_dict[cluster].remove(chosen_element)
            subset_documents = [
                retrieved_points[idx]["content"] for idx in subset
            ]
            subsets.append(subset_documents)

        return subsets



def bm25search_search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, Tuple[str, str]], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
    # Index the corpus within elastic-search
    # False, if the corpus has been already indexed
    if self.initialize:
        self.index(corpus)
        # Sleep for few seconds so that elastic-search indexes the docs properly
        time.sleep(self.sleep_for)

    #retrieve results from BM25
    query_ids = list(queries.keys())
    filter_ids = [queries[qid][1] for qid in query_ids]
    queries = [queries[qid][0] for qid in query_ids]

    final_results: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que', disable=kwargs.get('disable_tqdm', False)):
        query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
        results = self.es.lexical_multisearch(
            texts=queries[start_idx:start_idx+self.batch_size],
            filter_ids=filter_ids[start_idx:start_idx+self.batch_size],
            top_hits=top_k)
        for (query_id, hit) in zip(query_ids_batch, results):
            scores = {}
            for corpus_id, score, text in hit['hits']:
                scores[corpus_id] = (score, text)
                final_results[query_id] = scores

    return final_results

BM25Search.search = bm25search_search


def elasticsearch_lexical_multisearch(self, texts: List[str], filter_ids: List[str] = None, top_hits: int = 10, skip: int = 0) -> Dict[str, object]:
    """Multiple Query search in Elasticsearch

    Args:
        texts (List[str]): Multiple query texts
        top_hits (int): top k hits to be retrieved
        skip (int, optional): top hits to be skipped. Defaults to 0.

    Returns:
        Dict[str, object]: Hit results
    """
    request = []

    assert skip + top_hits <= 10000, "Elastic-Search Window too large, Max-Size = 10000"

    filter_ids = filter_ids or ([None] * len(texts))
    for text, fid in zip(texts, filter_ids):
        req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
        if fid is not None:
            req_body = {
                "_source": True, # No need to return source objects
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": text,  # matching query with both text and title fields
                                "type": "best_fields",
                                "fields": [self.title_key, self.text_key],
                                "tie_breaker": 0.5
                            },
                        },
                        "filter": {
                            "term": {
                                "_id": fid
                            }
                        }
                    },
                },
                "size": skip + top_hits, # The same paragraph will occur in results
            }
        else:
            req_body = {
                "_source": True, # No need to return source objects
                "query": {
                    "multi_match": {
                        "query": text, # matching query with both text and title fields
                        "type": "best_fields",
                        "fields": [self.title_key, self.text_key],
                        "tie_breaker": 0.5
                    }
                },
                "size": skip + top_hits, # The same paragraph will occur in results
            }
        request.extend([req_head, req_body])

    res = self.es.msearch(body = request)

    result = []
    for resp in res["responses"]:
        responses = resp["hits"]["hits"][skip:] if 'hits' in resp else []

        hits = []
        for hit in responses:
            hits.append((hit["_id"], hit['_score'], hit['_source']['txt']))

        result.append(self.hit_template(es_res=resp, hits=hits))
    return result

ElasticSearch.lexical_multisearch = elasticsearch_lexical_multisearch


def elasticsearch_hit_template(self, es_res: Dict[str, object], hits: List[Tuple[str, float]]) -> Dict[str, object]:
    """Hit output results template

    Args:
        es_res (Dict[str, object]): Elasticsearch response
        hits (List[Tuple[str, float]]): Hits from Elasticsearch

    Returns:
        Dict[str, object]: Hit results
    """
    result = {
        'meta': {
            'total': es_res['hits']['total']['value'] if 'hits' in es_res else None,
            'took': es_res['took'] if 'took' in es_res else None,
            'num_hits': len(hits)
        },
        'hits': hits,
    }
    return result

ElasticSearch.hit_template = elasticsearch_hit_template