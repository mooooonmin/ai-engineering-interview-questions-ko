# 코딩과 실전 구현 (Coding and Practical Implementation)

[← 목차로 돌아가기](../README.md)

> 질문을 클릭하면 답변이 펼쳐집니다. 먼저 스스로 답해 본 뒤 펼쳐 보세요.  
> 답변은 학습용으로 새로 작성한 설명입니다. 원작자의 영문 자료는 각 답변 끝의 "더 읽어보기"에 있습니다.

<details>
<summary><b>Q1. 임베딩 모델과 벡터 데이터베이스를 사용해 기본 RAG 파이프라인을 구현하세요.</b></summary>

**구성:** 문서 청킹 → 임베딩 → 벡터 저장 → 질문 임베딩 → 유사 청크 검색 → 프롬프트 구성 → LLM 생성.

```python
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.Client()
col = db.create_collection("docs", metadata={"hnsw:space": "cosine"})

def chunk(text, size=500, overlap=50):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]

def index(docs: dict[str, str]):
    for doc_id, text in docs.items():
        chunks = chunk(text)
        col.add(
            ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embedder.encode(chunks, normalize_embeddings=True).tolist(),
            metadatas=[{"doc_id": doc_id, "chunk": i} for i in range(len(chunks))],
        )

def retrieve(question, k=4):
    q = embedder.encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))

def answer(question, llm):
    hits = retrieve(question)
    context = "\n\n".join(f"[{i+1}] (출처: {m['doc_id']})\n{d}" for i, (d, m) in enumerate(hits))
    prompt = f"""아래 문서만 근거로 답하세요. 근거가 없으면 "문서에서 찾을 수 없습니다"라고 답하세요.
답변 문장마다 [번호]로 출처를 표시하세요.

{context}

질문: {question}"""
    return llm(prompt)
```

**면접에서 함께 말할 개선점**

- 청킹: 단어 수 대신 토큰 수·문단 경계 기반 재귀 청킹
- 검색: BM25 결합(하이브리드), 리랭커 추가
- 메타데이터 필터(권한, 날짜), 증분 인덱싱
- 평가: 검색 recall@k, 답변 충실도

</details>

<details>
<summary><b>Q2. 도구 사용(예: 계산기, 웹 검색)이 가능한 간단한 AI 에이전트를 만드세요.</b></summary>

**ReAct 루프를 함수 호출 API로 구현** 합니다. 모델은 도구를 "요청"만 하고, 실행은 코드가 합니다.

```python
import json, ast, operator

def calculator(expression: str) -> str:
    # eval 대신 안전한 산술식 평가
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp): return ops[type(n.op)](ev(n.operand))
        raise ValueError("허용되지 않는 식")
    return str(ev(ast.parse(expression, mode="eval").body))

def web_search(query: str) -> str:
    return search_api(query)[:2000]          # 실제 검색 API 호출, 결과 길이 제한

TOOLS = {"calculator": calculator, "web_search": web_search}
TOOL_SPECS = [
    {"name": "calculator", "description": "산술식을 정확히 계산합니다. 계산이 필요하면 반드시 사용하세요.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    {"name": "web_search", "description": "최신 정보나 모르는 사실을 웹에서 검색합니다.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]

def run_agent(question, llm, max_steps=8):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = llm(messages=messages, tools=TOOL_SPECS)       # 모델 호출
        messages.append({"role": "assistant", "content": resp.content})
        if not resp.tool_calls:                                # 도구 호출 없음 → 최종 답
            return resp.text
        for call in resp.tool_calls:
            try:
                result = TOOLS[call.name](**call.arguments)
            except Exception as e:
                result = f"오류: {e}"                          # 오류도 모델에 알려 스스로 복구
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "단계 한도에 도달했습니다. 지금까지 결과: ..."
```

**핵심 포인트**

- 도구 설명이 도구 선택 정확도를 좌우합니다.
- 최대 단계 수, 오류를 관찰로 돌려주기, 도구 결과 길이 제한.
- `eval()` 같은 위험한 실행 금지, 도구 인자 검증.

**더 읽어보기 (영문):** [ReAct Agent](https://outcomeschool.com/blog/react-agent)

</details>

<details>
<summary><b>Q3. 임베딩과 코사인 유사도를 사용해 의미 검색을 구현하세요.</b></summary>

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs: list[str] = []
        self.matrix = None                      # (N, d), 정규화된 임베딩

    def add(self, docs: list[str]):
        vecs = self.model.encode(docs, normalize_embeddings=True)
        self.docs.extend(docs)
        self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])

    def search(self, query: str, k: int = 5, min_score: float = 0.0):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q                # 정규화된 벡터 → 내적 = 코사인 유사도
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]          # 상위 k를 O(N)으로 선택
        top = top[np.argsort(-scores[top])]                # 그 안에서만 정렬
        return [(self.docs[i], float(scores[i])) for i in top if scores[i] >= min_score]

s = SemanticSearch()
s.add(["환불은 구매 후 7일 이내 가능합니다.", "배송은 2~3일 걸립니다.", "회원 등급 안내"])
print(s.search("돈 돌려받으려면 언제까지 신청해야 해?"))
```

**설명**

- 벡터를 길이 1로 정규화하면 코사인 유사도가 단순 내적이 되어 행렬곱 한 번으로 모든 문서 점수를 계산합니다.
- `argpartition`으로 전체 정렬 없이 상위 k개를 찾습니다.

**확장 시 고려**

- 문서가 수십만 개 이상이면 FAISS(HNSW, IVF) 같은 ANN 인덱스나 벡터 DB 사용.
- 비대칭 검색 모델은 질문·문서 접두어 규칙을 지킵니다.
- 정확한 키워드가 중요한 경우 BM25와 결합합니다.

**더 읽어보기 (영문):** [How does Semantic Search work?](https://outcomeschool.com/blog/how-does-semantic-search-work), [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)

</details>

<details>
<summary><b>Q4. 여러 가지 텍스트 청킹 전략(고정 크기, 재귀, 의미 기반)을 코드로 작성하세요.</b></summary>

```python
import re
import numpy as np

# 1) 고정 크기 (토큰 기준 + 겹침)
def fixed_chunks(tokens: list[str], size=300, overlap=50):
    step = size - overlap
    return [tokens[i:i + size] for i in range(0, max(len(tokens) - overlap, 1), step)]

# 2) 재귀 청킹: 큰 구분자부터 시도, 너무 크면 더 작은 구분자로
SEPARATORS = ["\n\n", "\n", ". ", " "]

def recursive_chunks(text: str, max_len=1000, seps=SEPARATORS):
    if len(text) <= max_len:
        return [text]
    if not seps:                                     # 더 쪼갤 구분자가 없으면 강제로 자름
        return [text[i:i + max_len] for i in range(0, len(text), max_len)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    parts = [p + sep for p in parts[:-1]] + [parts[-1]]   # 구분자를 앞 조각에 붙여 보존
    chunks, buf = [], ""
    for p in parts:
        if len(buf) + len(p) <= max_len:
            buf += p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_len:                     # 조각 하나가 너무 크면 더 작은 구분자로 재귀
                chunks.extend(recursive_chunks(p, max_len, rest)); buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]

# 3) 의미 기반 청킹: 인접 문장 유사도가 급락하는 지점에서 자름
def semantic_chunks(text: str, embed, threshold_pct=20, max_sentences=20):
    sents = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", text) if s.strip()]
    if len(sents) < 2:
        return sents
    vecs = np.array(embed(sents))                    # 정규화된 문장 임베딩
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)        # 인접 문장 코사인 유사도
    cut = np.percentile(sims, threshold_pct)         # 하위 20% 지점을 경계로
    chunks, cur = [], [sents[0]]
    for i, s in enumerate(sents[1:]):
        if sims[i] < cut or len(cur) >= max_sentences:
            chunks.append(" ".join(cur)); cur = []
        cur.append(s)
    chunks.append(" ".join(cur))
    return chunks
```

**선택 가이드:** 재귀 청킹을 기본으로, 문서 구조(마크다운 제목, 코드 함수)가 있으면 구조 기반 청킹을 우선하고, 의미 기반은 평가로 이득이 확인될 때 사용합니다. 실제로는 글자 수 대신 토크나이저로 토큰 수를 셉니다.

**더 읽어보기 (영문):** [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)

</details>

<details>
<summary><b>Q5. 변수 치환을 지원하는 프롬프트 템플릿 시스템을 구현하세요.</b></summary>

```python
import re
from dataclasses import dataclass, field

@dataclass
class PromptTemplate:
    name: str
    version: str
    template: str
    required: set[str] = field(default_factory=set)
    defaults: dict = field(default_factory=dict)
    max_len: dict = field(default_factory=dict)       # 변수별 최대 글자 수

    VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")          # {{ name }} 문법

    def variables(self) -> set[str]:
        return set(self.VAR.findall(self.template))

    def render(self, **values) -> str:
        vals = {**self.defaults, **values}
        missing = (self.required | self.variables()) - vals.keys()
        if missing:
            raise ValueError(f"[{self.name}@{self.version}] 누락된 변수: {sorted(missing)}")
        def sub(m):
            key = m.group(1)
            v = str(vals[key])
            if key in self.max_len:
                v = v[: self.max_len[key]]
            return v                                   # 값 안의 {{ }}는 다시 치환되지 않음
        return self.VAR.sub(sub, self.template)

class PromptRegistry:
    def __init__(self):
        self._store: dict[tuple[str, str], PromptTemplate] = {}
        self._labels: dict[tuple[str, str], str] = {}  # (name, label) -> version

    def register(self, t: PromptTemplate, label: str | None = None):
        self._store[(t.name, t.version)] = t
        if label:
            self._labels[(t.name, label)] = t.version

    def get(self, name: str, label: str = "production") -> PromptTemplate:
        return self._store[(name, self._labels[(name, label)])]

reg = PromptRegistry()
reg.register(PromptTemplate(
    name="support", version="v3",
    template="당신은 {{company}} 상담원입니다.\n<docs>\n{{context}}\n</docs>\n질문: {{question}}",
    defaults={"company": "ACME"}, max_len={"context": 8000},
), label="production")

prompt = reg.get("support").render(context=docs_text, question=user_q)
```

**설계 포인트**

- 누락 변수는 조용히 넘어가지 않고 오류로 처리합니다.
- 값 안의 템플릿 문법이 다시 해석되지 않게 한 번만 치환합니다(주입 방지).
- 버전과 라벨(production, staging)로 배포·롤백, 로그에는 이름@버전을 기록합니다.
- 고정 부분을 앞에, 변수 부분을 뒤에 두어 프롬프트 캐싱 효율을 높입니다.

</details>

<details>
<summary><b>Q6. LLM-as-a-judge를 사용해 LLM 출력 평가 파이프라인을 만드세요.</b></summary>

```python
import json, statistics
from concurrent.futures import ThreadPoolExecutor

JUDGE_PROMPT = """당신은 엄격한 평가자입니다. 아래 기준으로 답변을 평가하세요.

[질문]
{question}

[참고 정답 (있다면)]
{reference}

[평가할 답변]
{answer}

기준:
- correctness (1~5): 사실과 정답에 부합하는가
- relevance (1~5): 질문에 직접 답하는가
- faithfulness (1~5): 근거 없는 주장이 없는가

먼저 각 기준의 근거를 짧게 쓰고, 마지막에 JSON만 출력하세요:
{{"correctness": n, "relevance": n, "faithfulness": n, "reason": "..."}}"""

def judge_one(judge_llm, case, answer):
    raw = judge_llm(JUDGE_PROMPT.format(question=case["question"],
                                        reference=case.get("reference", "없음"), answer=answer),
                    temperature=0)
    data = json.loads(raw[raw.rfind("{"): raw.rfind("}") + 1])
    return {**data, "id": case["id"]}

def run_eval(cases, system_under_test, judge_llm, threshold=4.0):
    def task(case):
        ans = system_under_test(case["question"])
        return judge_one(judge_llm, case, ans) | {"answer": ans}
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(task, cases))
    summary = {k: statistics.mean(r[k] for r in results)
               for k in ("correctness", "relevance", "faithfulness")}
    failures = [r for r in results if min(r["correctness"], r["faithfulness"]) < threshold]
    return summary, failures, results
```

**신뢰성을 위한 장치**

- 판정자는 온도 0, 근거를 먼저 쓰고 점수를 내게 해 일관성 향상.
- 판정 프롬프트·판정 모델도 버전 관리.
- **사람 라벨 샘플로 판정자 검증:** 판정 점수와 사람 점수의 일치도(상관, 정확도)를 측정.
- 쌍 비교라면 순서를 바꿔 두 번 평가해 위치 편향 제거.
- 결정적 검사(형식, 금지어)는 판정자 대신 코드로.
- 결과를 이전 실행과 비교해 회귀 사례를 리포트.

**더 읽어보기 (영문):** [LLM as a Judge](https://outcomeschool.com/blog/llm-as-a-judge)

</details>

<details>
<summary><b>Q7. LLM API의 스트리밍 응답을 구현하세요.</b></summary>

**서버 (FastAPI, SSE)**

```python
import json, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/v1/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()

    async def events():
        full = []
        try:
            async for delta in llm_stream(body["messages"]):      # 제공사 SDK의 스트리밍 이터레이터
                if await req.is_disconnected():                     # 클라이언트가 끊으면 생성 중단
                    break
                full.append(delta)
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            log_completion("".join(full))                           # 완료 후 전체 응답 기록

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

**클라이언트 (Python)**

```python
import httpx, json

async def consume(messages):
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", URL, json={"messages": messages}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    print(json.loads(payload)["delta"], end="", flush=True)
```

**주의할 점**

- 프록시(Nginx)의 버퍼링을 끄고(`X-Accel-Buffering: no`) 타임아웃을 늘립니다.
- 클라이언트 연결이 끊기면 LLM 생성을 취소해 비용 낭비를 막습니다.
- 오류를 스트림 이벤트로 알리고, 종료 이벤트를 명확히 보냅니다.
- 출력 가드레일이 필요하면 문장 단위로 버퍼링 후 검사해 전송합니다.

**더 읽어보기 (영문):** [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)

</details>

<details>
<summary><b>Q8. 간단한 벡터 유사도 검색을 처음부터 직접 만드세요.</b></summary>

라이브러리 없이 **정확 검색(brute force)** 과 간단한 **근사 검색(IVF)** 을 구현합니다.

```python
import numpy as np

class FlatIndex:
    """전수 비교: 정확하지만 O(N·d)"""
    def __init__(self, dim):
        self.vecs = np.empty((0, dim), dtype=np.float32)
        self.ids: list = []

    def add(self, ids, vecs):
        v = np.asarray(vecs, dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12      # 정규화 → 내적 = 코사인
        self.vecs = np.vstack([self.vecs, v]); self.ids += list(ids)

    def search(self, q, k=5):
        q = np.asarray(q, dtype=np.float32); q /= np.linalg.norm(q) + 1e-12
        scores = self.vecs @ q
        top = np.argpartition(-scores, min(k, len(scores)) - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

class IVFIndex:
    """k-means로 군집을 만들고, 질의와 가까운 nprobe개 군집만 탐색하는 근사 검색"""
    def __init__(self, dim, n_lists=100, nprobe=8, iters=10):
        self.n_lists, self.nprobe, self.iters = n_lists, nprobe, iters
        self.flat = FlatIndex(dim)

    def build(self, ids, vecs):
        self.flat.add(ids, vecs)
        X = self.flat.vecs
        rng = np.random.default_rng(0)
        C = X[rng.choice(len(X), self.n_lists, replace=False)]
        for _ in range(self.iters):                                   # 간단한 k-means
            assign = np.argmax(X @ C.T, axis=1)
            for c in range(self.n_lists):
                m = X[assign == c]
                if len(m):
                    C[c] = m.mean(0) / (np.linalg.norm(m.mean(0)) + 1e-12)
        self.centroids, self.assign = C, np.argmax(X @ C.T, axis=1)

    def search(self, q, k=5):
        q = np.asarray(q, dtype=np.float32); q /= np.linalg.norm(q) + 1e-12
        lists = np.argsort(-(self.centroids @ q))[: self.nprobe]       # 가까운 군집 선택
        cand = np.where(np.isin(self.assign, lists))[0]
        scores = self.flat.vecs[cand] @ q
        order = np.argsort(-scores)[:k]
        return [(self.flat.ids[cand[i]], float(scores[i])) for i in order]
```

**설명**

- `nprobe`를 늘리면 재현율이 오르고 속도는 느려집니다(정확도-속도 트레이드오프).
- 실무에서는 FAISS, hnswlib, 벡터 DB를 사용하되, 이 원리를 이해하면 파라미터 튜닝과 문제 진단이 쉬워집니다.

**더 읽어보기 (영문):** [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)

</details>

<details>
<summary><b>Q9. 챗봇을 위한 대화 메모리 시스템(슬라이딩 윈도우, 요약, 버퍼)을 구현하세요.</b></summary>

```python
from dataclasses import dataclass, field

def count_tokens(text: str) -> int:
    return max(1, len(text) // 3)          # 실제로는 모델 토크나이저 사용

@dataclass
class BufferMemory:
    """전체 대화 보관 (짧은 대화용)"""
    messages: list = field(default_factory=list)
    def add(self, role, content): self.messages.append({"role": role, "content": content})
    def context(self): return list(self.messages)

@dataclass
class SlidingWindowMemory(BufferMemory):
    """최근 N턴만 유지"""
    max_turns: int = 10
    def context(self): return self.messages[-self.max_turns * 2:]

@dataclass
class SummaryBufferMemory(BufferMemory):
    """토큰 예산을 넘으면 오래된 메시지를 요약으로 압축하고 최근 메시지는 원문 유지"""
    llm: object = None
    max_tokens: int = 3000
    keep_recent: int = 6
    summary: str = ""

    def add(self, role, content):
        super().add(role, content)
        total = count_tokens(self.summary) + sum(count_tokens(m["content"]) for m in self.messages)
        if total > self.max_tokens and len(self.messages) > self.keep_recent:
            old, self.messages = self.messages[:-self.keep_recent], self.messages[-self.keep_recent:]
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
            self.summary = self.llm(
                "기존 요약과 새 대화를 합쳐 갱신된 요약을 쓰세요. 사용자 이름·선호·결정 사항·"
                "숫자·진행 중인 요청은 반드시 보존하세요.\n"
                f"[기존 요약]\n{self.summary}\n[새 대화]\n{transcript}"
            )

    def context(self):
        head = [{"role": "system", "content": f"이전 대화 요약: {self.summary}"}] if self.summary else []
        return head + list(self.messages)

# 사용
mem = SummaryBufferMemory(llm=call_llm, max_tokens=2000)
mem.add("user", "제 이름은 민수예요. 다음 주 부산 여행 계획 중이에요.")
reply = call_llm(messages=[SYSTEM] + mem.context() + [{"role": "user", "content": new_msg}])
```

**선택 가이드**

- 버퍼: 짧은 대화, 최대 정확성
- 슬라이딩 윈도우: 단순하지만 오래된 정보 유실
- 요약 + 버퍼: 긴 대화의 표준적 절충
- 추가로 핵심 사실 추출(키-값)과 벡터 검색 기반 장기 메모리를 결합할 수 있습니다.

**더 읽어보기 (영문):** [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)

</details>

<details>
<summary><b>Q10. LLM 출력의 환각을 감지하고 처리하는 코드를 작성하세요.</b></summary>

**전략:** (1) 답변을 주장 단위로 나누고 (2) 각 주장이 근거 문서로 뒷받침되는지 판정하고 (3) 근거 없는 주장을 처리합니다. 보조로 자기 일관성 검사를 사용합니다.

```python
import json

EXTRACT = """다음 답변을 독립적으로 검증 가능한 짧은 사실 주장 목록으로 나누세요.
JSON 배열만 출력하세요.\n답변:\n{answer}"""

VERIFY = """[근거 문서]
{context}

[주장]
{claim}

근거 문서가 이 주장을 뒷받침하면 "supported", 모순되면 "contradicted",
판단할 정보가 없으면 "not_found"로 답하고, 근거 문장을 인용하세요.
JSON으로만: {{"label": "...", "evidence": "..."}}"""

def detect_hallucinations(answer, context, llm):
    claims = json.loads(llm(EXTRACT.format(answer=answer), temperature=0))
    results = []
    for c in claims:
        r = json.loads(llm(VERIFY.format(context=context, claim=c), temperature=0))
        results.append({"claim": c, **r})
    unsupported = [r for r in results if r["label"] != "supported"]
    score = 1 - len(unsupported) / max(len(results), 1)          # 충실도 점수
    return score, unsupported, results

def self_consistency(question, llm, n=5):
    """여러 번 샘플링해 답이 일관된지 확인 (근거 문서가 없을 때의 보조 신호)"""
    answers = [llm(question, temperature=0.8) for _ in range(n)]
    verdict = llm("다음 답변들이 핵심 사실에서 서로 일치하는지 0~1 점수로만 답하세요:\n"
                  + "\n---\n".join(answers), temperature=0)
    return float(verdict)

def guarded_answer(question, context, llm, min_score=0.9):
    answer = llm(f"문서에 근거해서만 답하세요.\n{context}\n질문: {question}", temperature=0)
    score, bad, _ = detect_hallucinations(answer, context, llm)
    if score >= min_score:
        return answer
    if any(b["label"] == "contradicted" for b in bad) or score < 0.5:
        return "제공된 문서로는 확실한 답을 드리기 어렵습니다."
    # 근거 없는 주장만 제거하도록 재작성
    bad_list = "\n".join(b["claim"] for b in bad)
    return llm(f"다음 답변에서 아래 주장들을 제거하고 다시 쓰세요.\n답변: {answer}\n제거할 주장:\n{bad_list}")
```

**보완:** 판정에 NLI 모델(빠르고 저렴)을 먼저 쓰고 애매한 경우만 LLM 판정, 숫자·고유명사는 원문 문자열 매칭으로 추가 검사, 결과를 로그로 남겨 환각률 모니터링.

</details>

<details>
<summary><b>Q11. LLM API 호출에 지수 백오프(exponential backoff)를 적용한 재시도 메커니즘을 구현하세요.</b></summary>

```python
import random, time, asyncio
from functools import wraps

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

class RetryableError(Exception):
    def __init__(self, msg, retry_after: float | None = None):
        super().__init__(msg); self.retry_after = retry_after

def with_retry(max_attempts=5, base=0.5, cap=30.0, timeout_budget=60.0):
    def deco(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.monotonic()
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except RetryableError as e:
                    if attempt == max_attempts:
                        raise
                    # 지수 백오프 + 풀 지터 (재시도가 한 시점에 몰리는 것을 방지)
                    delay = random.uniform(0, min(cap, base * 2 ** (attempt - 1)))
                    if e.retry_after:                       # 서버가 알려 준 대기 시간 우선
                        delay = max(delay, e.retry_after)
                    if time.monotonic() - start + delay > timeout_budget:
                        raise                               # 전체 시간 예산 초과 시 중단
                    await asyncio.sleep(delay)
        return wrapper
    return deco

@with_retry(max_attempts=5)
async def call_llm(client, payload):
    resp = await client.post("/v1/messages", json=payload, timeout=30)
    if resp.status_code in RETRYABLE_STATUS:
        ra = resp.headers.get("retry-after")
        raise RetryableError(f"status {resp.status_code}", float(ra) if ra else None)
    resp.raise_for_status()                                 # 400, 401 등은 재시도하지 않음
    return resp.json()
```

**포인트**

- **재시도할 오류만 재시도:** 429, 5xx, 타임아웃은 재시도, 400(잘못된 요청)·401(인증)은 즉시 실패.
- **지터:** 많은 클라이언트가 동시에 재시도해 부하가 다시 몰리는 것(thundering herd) 방지.
- `Retry-After` 헤더 존중, 전체 시간 예산과 최대 횟수 제한.
- 스트리밍 도중 실패, 부작용이 있는 요청(도구 실행)은 멱등성을 확인한 뒤 재시도.
- 반복 실패 시 서킷 브레이커·폴백 모델로 넘어가도록 상위 계층과 결합합니다.

</details>

<details>
<summary><b>Q12. LLM API를 위한 함수 호출(도구 사용) 핸들러를 작성하세요.</b></summary>

```python
import inspect, json, typing
from pydantic import create_model, ValidationError

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def tool(self, description: str):
        """함수 시그니처에서 JSON 스키마를 자동 생성하는 데코레이터"""
        def deco(fn):
            sig = inspect.signature(fn)
            hints = typing.get_type_hints(fn)
            fields = {n: (hints[n], ... if p.default is p.empty else p.default)
                      for n, p in sig.parameters.items()}
            model = create_model(f"{fn.__name__}_args", **fields)
            self.tools[fn.__name__] = {"fn": fn, "model": model, "spec": {
                "name": fn.__name__, "description": description,
                "input_schema": model.model_json_schema()}}
            return fn
        return deco

    def specs(self):
        return [t["spec"] for t in self.tools.values()]

    def execute(self, name: str, arguments: dict | str) -> str:
        if name not in self.tools:
            return f"오류: '{name}' 도구는 없습니다. 사용 가능: {list(self.tools)}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return "오류: 인자가 올바른 JSON이 아닙니다."
        try:
            args = self.tools[name]["model"](**arguments)             # 타입·필수값 검증
        except ValidationError as e:
            return f"인자 오류: {e.errors()[0]['msg']} — 스키마를 확인하고 다시 호출하세요."
        try:
            result = self.tools[name]["fn"](**args.model_dump())
            text = json.dumps(result, ensure_ascii=False, default=str)
            return text[:4000]                                         # 결과 길이 제한
        except Exception as e:
            return f"실행 오류: {e}"

registry = ToolRegistry()

@registry.tool("주문 번호로 주문 상태를 조회합니다. 주문 번호 형식: ORD-12345")
def get_order_status(order_id: str) -> dict:
    return db.lookup(order_id)

# 모델 응답 처리
def handle(resp, messages):
    for call in resp.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": registry.execute(call.name, call.arguments)})
```

**핵심:** 스키마 자동 생성, 서버 측 인자 검증(모델 출력은 신뢰할 수 없는 입력), 모델이 스스로 고칠 수 있는 오류 메시지, 결과 길이 제한, 쓰기·삭제 같은 위험 도구는 권한 검사와 사람 확인을 추가합니다.

**더 읽어보기 (영문):** [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)

</details>

<details>
<summary><b>Q13. 검색 결과를 위한 간단한 리랭커를 구현하세요.</b></summary>

**방법 1: 크로스 인코더 리랭커 (실무 표준)**

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model="BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model)

    def rerank(self, query: str, docs: list[str], top_k=5, min_score=None):
        scores = self.model.predict([(query, d) for d in docs], batch_size=32)
        ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        if min_score is not None:
            ranked = [r for r in ranked if r[1] >= min_score]
        return ranked[:top_k]
```

**방법 2: 여러 신호를 결합한 가벼운 리랭커 (모델 없이)**

```python
import math, re
from collections import Counter

def tokenize(t): return re.findall(r"\w+", t.lower())

def feature_rerank(query, hits, w_sem=0.6, w_kw=0.3, w_fresh=0.1, now_days=0):
    """hits: [{"text":..., "sem_score":..., "age_days":...}]"""
    q = Counter(tokenize(query))
    out = []
    for h in hits:
        d = Counter(tokenize(h["text"]))
        overlap = sum(min(q[t], d[t]) for t in q) / max(sum(q.values()), 1)   # 키워드 일치율
        fresh = math.exp(-h.get("age_days", 0) / 365)                            # 최신성
        score = w_sem * h["sem_score"] + w_kw * overlap + w_fresh * fresh
        out.append({**h, "score": score})
    return sorted(out, key=lambda x: -x["score"])
```

**방법 3: LLM 리랭커 (소수 후보, 높은 정확도)**

- 후보 10개 정도를 번호와 함께 보여 주고 "질문에 답하는 데 유용한 순서로 번호를 나열하라"고 요청합니다. 비용이 크므로 크로스 인코더 이후 마지막 단계에서만 사용합니다.

**사용 흐름:** 1단계 검색(하이브리드)으로 50~100개 → 크로스 인코더로 상위 5~10개 → LLM에 전달. 리랭커 점수에 임계값을 두면 "관련 문서 없음" 판단에도 쓸 수 있습니다.

**더 읽어보기 (영문):** [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work)

</details>

<details>
<summary><b>Q14. PDF에서 텍스트를 추출해 청크로 나누는 기본 문서 파서를 만드세요.</b></summary>

```python
import re
import fitz                                  # PyMuPDF

def extract_pages(path: str):
    """페이지별 텍스트 블록을 읽는 순서대로 추출"""
    doc = fitz.open(path)
    pages = []
    for pno, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")                       # (x0, y0, x1, y1, text, ...)
        blocks = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))   # 위→아래, 왼→오른쪽
        text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
        pages.append({"page": pno, "text": text})
    return pages

def remove_repeated_lines(pages, min_ratio=0.6):
    """여러 페이지에 반복되는 머리말·꼬리말 제거"""
    from collections import Counter
    counts = Counter(line for p in pages for line in set(p["text"].splitlines()))
    repeated = {l for l, c in counts.items() if c >= max(2, min_ratio * len(pages))}
    for p in pages:
        p["text"] = "\n".join(l for l in p["text"].splitlines()
                              if l not in repeated and not re.fullmatch(r"\s*\d+\s*", l))
    return pages

def clean(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)               # 줄바꿈으로 끊긴 하이픈 단어 복원
    text = re.sub(r"(?<![.!?:])\n(?!\n)", " ", text)           # 문장 중간 줄바꿈 합치기
    return re.sub(r"[ \t]+", " ", text).strip()

def chunk_pdf(path, max_chars=1500, overlap=200):
    pages = remove_repeated_lines(extract_pages(path))
    chunks, buf, start_page = [], "", 1
    for p in pages:
        for para in re.split(r"\n{2,}", clean(p["text"])):
            if len(buf) + len(para) > max_chars and buf:
                chunks.append({"text": buf, "page_start": start_page, "page_end": p["page"],
                               "source": path})
                buf, start_page = buf[-overlap:], p["page"]     # 겹침 유지
            buf = (buf + "\n\n" + para).strip()
    if buf:
        chunks.append({"text": buf, "page_start": start_page, "page_end": pages[-1]["page"],
                       "source": path})
    return chunks
```

**한계와 확장**

- 스캔 PDF(텍스트 레이어 없음)는 OCR 필요(Tesseract, PaddleOCR) — `page.get_text()`가 비어 있으면 OCR로 전환.
- 표·다단 레이아웃이 복잡하면 레이아웃 인식 파서(Docling, Unstructured, Marker)나 VLM 사용.
- 청크에 페이지 번호·섹션 제목을 메타데이터로 저장해 출처 표시에 활용합니다.

**더 읽어보기 (영문):** [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)

</details>

<details>
<summary><b>Q15. 코사인 유사도, 내적, 유클리드 거리 함수를 처음부터 직접 구현하세요.</b></summary>

```python
import math

def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("차원이 다릅니다")
    return sum(x * y for x, y in zip(a, b))

def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))

def cosine_similarity(a, b, eps=1e-12) -> float:
    return dot(a, b) / (max(norm(a) * norm(b), eps))      # 0 벡터 나눗셈 방지

def euclidean_distance(a, b) -> float:
    if len(a) != len(b):
        raise ValueError("차원이 다릅니다")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# NumPy 벡터화 버전 (질의 1개 vs 문서 N개)
import numpy as np

def cosine_matrix(q: np.ndarray, M: np.ndarray, eps=1e-12) -> np.ndarray:
    return (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + eps)

def euclidean_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    # ||m - q||² = ||m||² - 2 m·q + ||q||²  (큰 행렬에서 메모리 효율적)
    sq = (M ** 2).sum(1) - 2 * (M @ q) + (q ** 2).sum()
    return np.sqrt(np.maximum(sq, 0))                       # 부동소수 오차로 생긴 음수 방지

a, b = [1.0, 2.0, 3.0], [2.0, 4.0, 6.5]
print(dot(a, b), cosine_similarity(a, b), euclidean_distance(a, b))
```

**관계 정리**

- 벡터를 길이 1로 정규화하면 `cosine = dot`, `euclidean² = 2 - 2·cosine` → 세 지표 모두 같은 순위를 줍니다.
- 그래서 실무에서는 임베딩을 정규화해 두고 가장 빠른 내적을 씁니다.

**더 읽어보기 (영문):** [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)

</details>

<details>
<summary><b>Q16. 토큰 수 계산과 컨텍스트 윈도우 관리를 구현하는 코드를 작성하세요.</b></summary>

```python
import tiktoken

class ContextManager:
    def __init__(self, model_window=128_000, max_output=2_000, encoding="cl100k_base",
                 per_message_overhead=4):
        self.enc = tiktoken.get_encoding(encoding)   # 실제로는 사용하는 모델의 토크나이저
        self.budget = model_window - max_output       # 출력 공간을 먼저 확보
        self.overhead = per_message_overhead          # 역할 토큰 등 메시지당 추가 토큰(근사)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))

    def count_messages(self, messages) -> int:
        return sum(self.count(m["content"]) + self.overhead for m in messages)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        ids = self.enc.encode(text)
        return text if len(ids) <= max_tokens else self.enc.decode(ids[:max_tokens]) + " …(생략)"

    def build(self, system: str, history: list, docs: list[str], question: str,
              doc_share=0.5):
        """우선순위: 시스템 > 질문 > 문서(관련도 순) > 최근 대화"""
        fixed = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        remaining = self.budget - self.count_messages(fixed)
        if remaining < 0:
            raise ValueError("시스템 프롬프트와 질문만으로 예산 초과")

        # 1) 문서: 예산의 일부만 사용, 관련도 순으로 채우기
        doc_budget, ctx_docs = int(remaining * doc_share), []
        for d in docs:
            t = self.count(d)
            if t > doc_budget:
                if doc_budget > 200:
                    ctx_docs.append(self.truncate_text(d, doc_budget))
                break
            ctx_docs.append(d); doc_budget -= t
        doc_msg = {"role": "system", "content": "참고 문서:\n" + "\n\n".join(ctx_docs)}
        remaining -= self.count_messages([doc_msg])

        # 2) 대화 기록: 최신부터 거꾸로 채우기
        kept = []
        for m in reversed(history):
            t = self.count_messages([m])
            if t > remaining:
                break
            kept.insert(0, m); remaining -= t

        return [fixed[0], doc_msg, *kept, fixed[1]]
```

**포인트**

- 출력 토큰 공간(`max_output`)을 먼저 빼고 입력 예산을 계산합니다.
- 사용하는 모델의 토크나이저로 세야 정확합니다(모델마다 다름). 메시지 형식 오버헤드도 고려합니다.
- 잘라낼 때는 우선순위를 정하고, 버린 대화는 요약으로 대체하는 확장이 가능합니다.

</details>

<details>
<summary><b>Q17. 간단한 프롬프트 버전 관리 시스템을 만드세요.</b></summary>

```python
import hashlib, json, sqlite3, datetime as dt

class PromptStore:
    """불변 버전 + 라벨(production/staging) + 이력. SQLite 기반"""
    def __init__(self, path="prompts.db"):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS versions(
            name TEXT, version INTEGER, template TEXT, model TEXT, params TEXT,
            hash TEXT, author TEXT, note TEXT, created TEXT, PRIMARY KEY(name, version));
        CREATE TABLE IF NOT EXISTS labels(
            name TEXT, label TEXT, version INTEGER, updated TEXT, PRIMARY KEY(name, label));
        CREATE TABLE IF NOT EXISTS label_history(
            name TEXT, label TEXT, version INTEGER, changed TEXT, by TEXT);
        """)

    def create(self, name, template, model, params=None, author="", note=""):
        h = hashlib.sha256(f"{template}|{model}|{json.dumps(params, sort_keys=True)}".encode()).hexdigest()[:12]
        last = self.db.execute("SELECT version, hash FROM versions WHERE name=? ORDER BY version DESC LIMIT 1",
                               (name,)).fetchone()
        if last and last[1] == h:
            return last[0]                                  # 내용이 같으면 새 버전 만들지 않음
        v = (last[0] + 1) if last else 1
        self.db.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?)",
                        (name, v, template, model, json.dumps(params or {}), h, author, note,
                         dt.datetime.utcnow().isoformat()))
        self.db.commit()
        return v

    def set_label(self, name, label, version, by=""):
        now = dt.datetime.utcnow().isoformat()
        self.db.execute("INSERT OR REPLACE INTO labels VALUES(?,?,?,?)", (name, label, version, now))
        self.db.execute("INSERT INTO label_history VALUES(?,?,?,?,?)", (name, label, version, now, by))
        self.db.commit()

    def get(self, name, label="production"):
        row = self.db.execute("""SELECT v.version, v.template, v.model, v.params FROM labels l
            JOIN versions v ON v.name=l.name AND v.version=l.version
            WHERE l.name=? AND l.label=?""", (name, label)).fetchone()
        if not row:
            raise KeyError(f"{name}@{label} 없음")
        return {"version": row[0], "template": row[1], "model": row[2], "params": json.loads(row[3])}

    def rollback(self, name, label="production", by=""):
        hist = self.db.execute("""SELECT version FROM label_history WHERE name=? AND label=?
                                  ORDER BY changed DESC LIMIT 2""", (name, label)).fetchall()
        if len(hist) < 2:
            raise ValueError("되돌릴 이전 버전이 없습니다")
        self.set_label(name, label, hist[1][0], by=by)
```

**운영 방식**

- 새 버전은 먼저 `staging` 라벨 → 평가셋 통과 → `production` 라벨로 승격(CI 연동).
- 요청 로그에 `name@version`을 기록해 문제를 재현합니다.
- 앱은 라벨로 조회하고 결과를 캐시(짧은 TTL)해 저장소 장애에 대비합니다.
- 프롬프트와 함께 모델·파라미터를 한 버전으로 묶어 관리합니다.

</details>

<details>
<summary><b>Q18. LLM 응답을 위한 캐싱 레이어를 구현하세요.</b></summary>

```python
import hashlib, json, time

class LLMCache:
    """정확 일치 캐시. backend는 get/set(ttl) 인터페이스 (Redis 등)"""
    def __init__(self, backend, default_ttl=3600, namespace="llm"):
        self.b, self.ttl, self.ns = backend, default_ttl, namespace
        self.hits = self.misses = 0

    def key(self, model, messages, params, scope=""):
        payload = json.dumps({"m": model, "msg": messages, "p": params, "scope": scope},
                             sort_keys=True, ensure_ascii=False)
        return f"{self.ns}:{hashlib.sha256(payload.encode()).hexdigest()}"

    def cacheable(self, params) -> bool:
        # 결정적 설정일 때만 캐시 (온도 > 0이면 매번 다른 답이 기대되므로 제외)
        return params.get("temperature", 1.0) == 0 and not params.get("stream", False)

    def call(self, llm, model, messages, params, scope="", ttl=None):
        if not self.cacheable(params):
            return llm(model=model, messages=messages, **params)
        k = self.key(model, messages, params, scope)
        if (hit := self.b.get(k)) is not None:
            self.hits += 1
            return json.loads(hit)["response"]
        self.misses += 1
        resp = llm(model=model, messages=messages, **params)
        self.b.set(k, json.dumps({"response": resp, "at": time.time()}), ttl or self.ttl)
        return resp

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
```

**설계 포인트**

- **캐시 키에 모든 영향 요소 포함:** 모델 버전, 메시지 전체, 파라미터, 그리고 **권한 범위(scope: 사용자·테넌트)** — 다른 사용자 데이터가 섞인 응답이 재사용되지 않게.
- **결정적 요청만 캐시:** 온도 0, 비스트리밍.
- **TTL과 무효화:** 근거 데이터가 바뀔 수 있는 응답은 짧은 TTL, 프롬프트·모델 버전 변경 시 네임스페이스를 바꿔 일괄 무효화.
- **개인정보:** 민감한 응답은 캐시하지 않거나 암호화.
- **지표:** 적중률, 절감 토큰·비용.
- 의미가 비슷한 질문까지 재사용하려면 시맨틱 캐시(다음 문제)를 추가로 둡니다.

</details>

<details>
<summary><b>Q19. LLM 쿼리를 위한 시맨틱 캐싱(의미가 비슷한 쿼리의 응답을 캐시)을 구현하세요.</b></summary>

```python
import time
import numpy as np

class SemanticCache:
    def __init__(self, embed, threshold=0.92, ttl=3600, max_items=10_000):
        self.embed, self.th, self.ttl, self.max = embed, threshold, ttl, max_items
        self.vecs = np.empty((0, 0), dtype=np.float32)
        self.items: list[dict] = []

    def _norm(self, v):
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def lookup(self, query: str, scope: str):
        if not self.items:
            return None
        q = self._norm(self.embed(query))
        sims = self.vecs @ q
        now = time.time()
        for i in np.argsort(-sims)[:5]:                    # 상위 후보 몇 개만 확인
            it = self.items[i]
            if sims[i] < self.th:
                break
            if it["scope"] == scope and now - it["at"] < self.ttl:   # 권한 범위·만료 확인
                it["hits"] += 1
                return it["response"], float(sims[i])
        return None

    def store(self, query: str, response: str, scope: str):
        v = self._norm(self.embed(query))[None, :]
        self.vecs = v if self.vecs.size == 0 else np.vstack([self.vecs, v])
        self.items.append({"query": query, "response": response, "scope": scope,
                           "at": time.time(), "hits": 0})
        if len(self.items) > self.max:                     # 가장 오래된 항목 제거
            self.vecs, self.items = self.vecs[1:], self.items[1:]

def answer(query, user_scope, cache: SemanticCache, llm):
    if hit := cache.lookup(query, user_scope):
        return hit[0]
    resp = llm(query)
    if is_cacheable(query, resp):                          # 개인화·시간 민감 질문은 제외
        cache.store(query, resp, user_scope)
    return resp
```

**중요한 설계 결정**

- **임계값:** 너무 낮으면 다른 질문에 틀린 답을 재사용합니다("환불 기간은?" vs "교환 기간은?"). 질문 쌍 평가셋으로 정밀도를 측정해 정합니다.
- **캐시하면 안 되는 질문:** 개인 데이터("내 주문 상태"), 시간 민감("오늘 날씨"), 대화 맥락 의존 질문.
- **범위 분리:** 테넌트·권한·언어별로 분리.
- **확장:** 항목이 많으면 벡터 DB에 저장, TTL·무효화(근거 문서 변경 시).
- **검증 단계(선택):** 후보가 정말 같은 의도인지 작은 모델로 한 번 더 확인하면 오답 재사용을 줄일 수 있습니다.

**더 읽어보기 (영문):** [How does Semantic Caching work?](https://outcomeschool.com/blog/how-does-semantic-caching-work)

</details>

<details>
<summary><b>Q20. 사용자 입력에서 프롬프트 인젝션 시도를 감지하는 코드를 작성하세요.</b></summary>

**다층 탐지:** 빠른 규칙 → 분류 모델 → (애매하면) LLM 판정. 어떤 탐지도 완벽하지 않으므로 권한 최소화와 함께 써야 합니다.

```python
import re, base64, unicodedata

PATTERNS = [
    r"(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)",
    r"(이전|위의?|앞의?)\s*(모든\s*)?(지시|명령|규칙).{0,10}(무시|잊어)",
    r"(reveal|print|show|repeat).{0,20}(system\s*prompt|instructions)",
    r"시스템\s*프롬프트.{0,15}(보여|출력|알려)",
    r"you\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)\s*mode",
    r"</?(system|assistant)>|\[/?INST\]|<\|im_start\|>",          # 역할 토큰 흉내
]

def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)                       # 전각·변형 문자 정규화
    t = re.sub(r"[​-‏⁠﻿]", "", t)             # 보이지 않는 문자 제거
    return t.lower()

def decoded_variants(text: str):
    yield text
    for m in re.findall(r"[A-Za-z0-9+/=]{24,}", text):            # base64로 숨긴 지시
        try:
            yield base64.b64decode(m).decode("utf-8", "ignore")
        except Exception:
            pass

def rule_score(text: str) -> float:
    hits = sum(bool(re.search(p, normalize(v))) for v in decoded_variants(text) for p in PATTERNS)
    return min(1.0, hits * 0.5)

def detect_injection(text: str, classifier=None, llm=None) -> dict:
    score, reasons = rule_score(text), []
    if score:
        reasons.append("rule")
    if classifier:                                                 # 예: 인젝션 분류 모델
        p = classifier(text)
        score = max(score, p); reasons.append(f"clf={p:.2f}")
    if llm and 0.3 <= score < 0.8:                                 # 애매한 구간만 LLM 판정
        verdict = llm("다음 입력이 AI의 지시를 무시·변경하거나 숨겨진 정보를 빼내려는 시도인지 "
                      f"YES/NO로만 답하세요.\n입력:\n<<<{text}>>>", temperature=0)
        if verdict.strip().upper().startswith("YES"):
            score = max(score, 0.9); reasons.append("llm")
    return {"is_injection": score >= 0.8, "score": score, "reasons": reasons}
```

**사용 시 주의**

- 사용자 입력뿐 아니라 **도구 결과·검색 문서·이메일 같은 외부 콘텐츠** 에도 적용합니다(간접 인젝션).
- 탐지 시 조치: 차단, 경고 후 제한 모드(도구 비활성화), 로그·알림.
- 규칙은 쉽게 우회되므로 분류 모델과 레드 팀 데이터로 계속 갱신합니다.
- 최종 방어선은 탐지가 아니라 **최소 권한과 위험 행동 승인** 입니다.

**더 읽어보기 (영문):** [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)

</details>

<details>
<summary><b>Q21. 주제를 벗어난 응답과 PII 유출을 검사하는 LLM 출력 가드레일 시스템을 구현하세요.</b></summary>

```python
import re
from dataclasses import dataclass, field
import numpy as np

@dataclass
class GuardResult:
    allowed: bool
    text: str
    violations: list = field(default_factory=list)

PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone_kr": r"01[016789]-?\d{3,4}-?\d{4}",
    "rrn": r"\d{6}-?[1-4]\d{6}",                       # 주민등록번호 형식
    "card": r"\b(?:\d[ -]?){13,16}\b",
}

def luhn_ok(num: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", num)][::-1]
    return bool(d) and (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0

class OutputGuard:
    def __init__(self, embed, topic_examples: list[str], topic_threshold=0.35,
                 allowed_pii: set[str] | None = None):
        self.embed = embed
        self.topic = np.mean([self._n(embed(t)) for t in topic_examples], axis=0)
        self.topic = self._n(self.topic)
        self.th = topic_threshold
        self.allowed_pii = allowed_pii or set()          # 예: 사용자 본인 이메일

    @staticmethod
    def _n(v):
        v = np.asarray(v, dtype=np.float32); return v / (np.linalg.norm(v) + 1e-12)

    def check_topic(self, text):
        sim = float(self._n(self.embed(text)) @ self.topic)
        return sim >= self.th, sim

    def redact_pii(self, text):
        found = []
        def repl(kind):
            def f(m):
                val = m.group(0)
                if val in self.allowed_pii or (kind == "card" and not luhn_ok(val)):
                    return val
                found.append(kind); return f"[{kind.upper()} 가림]"
            return f
        for kind, pat in PII_PATTERNS.items():
            text = re.sub(pat, repl(kind), text)
        return text, found

    def __call__(self, response: str) -> GuardResult:
        violations = []
        on_topic, sim = self.check_topic(response)
        if not on_topic:
            violations.append(f"off_topic(sim={sim:.2f})")
            return GuardResult(False, "이 서비스에서는 해당 주제에 대해 답변드리기 어렵습니다.", violations)
        text, pii = self.redact_pii(response)
        violations += [f"pii:{k}" for k in pii]
        return GuardResult(True, text, violations)

guard = OutputGuard(embed, topic_examples=["주문, 배송, 환불, 교환에 대한 안내",
                                           "상품 사용법과 계정 문의"])
result = guard(llm_response)
log_guard(result.violations)
```

**확장**

- 주제 판단을 임베딩 유사도로 1차, 애매하면 작은 분류 모델·LLM 판정으로.
- 이름·주소 같은 문맥 PII는 NER 모델(Presidio 등) 추가.
- 스트리밍 출력은 문장 단위로 버퍼링해 검사.
- 위반 로그로 오탐·미탐을 분석해 임계값 조정.

**더 읽어보기 (영문):** [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)

</details>

<details>
<summary><b>Q22. 에이전트마다 역할이 다르고 함께 협력해 작업을 수행하는 멀티 에이전트 시스템을 만드세요.</b></summary>

**구조:** 오케스트레이터가 작업을 나누고, 역할별 에이전트(조사·작성·검토)가 구조화된 결과를 주고받습니다.

```python
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Agent:
    name: str
    system: str
    tools: list
    llm: callable

    def run(self, task: str, context: str = "", max_steps=6) -> str:
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": f"작업: {task}\n\n참고 자료:\n{context}"}]
        return run_tool_loop(self.llm, msgs, self.tools, max_steps)   # Q32의 에이전트 루프 재사용

researcher = Agent("researcher", "주어진 주제를 조사해 핵심 사실과 출처 URL을 JSON 목록으로 반환하라.",
                   [web_search], llm)
writer = Agent("writer", "조사 결과만 근거로 보고서를 작성하라. 각 주장에 출처를 표시하라.", [], llm)
reviewer = Agent("reviewer", "보고서를 조사 결과와 대조해 사실 오류·누락·근거 없는 주장을 찾아 "
                 'JSON으로 반환하라: {"approved": bool, "issues": [...]}', [], llm)

def orchestrate(goal: str, max_revisions=2):
    # 1) 계획: 하위 조사 주제 분해
    plan = json.loads(llm(f"다음 목표를 독립적인 조사 주제 3~5개로 나눠 JSON 배열로만 출력: {goal}"))
    # 2) 병렬 조사
    with ThreadPoolExecutor(max_workers=4) as ex:
        findings = list(ex.map(researcher.run, plan))
    research = "\n\n".join(f"## {t}\n{f}" for t, f in zip(plan, findings))
    # 3) 작성 → 검토 → 수정 반복 (상한 있음)
    draft = writer.run(goal, research)
    for _ in range(max_revisions):
        review = json.loads(reviewer.run("이 보고서를 검토하라", f"[조사]\n{research}\n\n[보고서]\n{draft}"))
        if review["approved"]:
            break
        draft = writer.run(f"{goal}\n다음 문제를 고쳐라: {review['issues']}", research + "\n\n[이전 초안]\n" + draft)
    return {"report": draft, "sources": research}
```

**설계 원칙**

- **역할마다 최소 도구·명확한 출력 형식:** 에이전트 간에는 자유 대화보다 구조화된 결과를 넘겨 정보 손실을 줄입니다.
- **독립 작업은 병렬:** 조사처럼 서로 의존하지 않는 작업.
- **반복 상한과 예산:** 검토-수정 루프, 토큰·시간 한도.
- **트레이싱:** 하나의 작업 ID로 모든 에이전트 호출을 추적.
- 단일 에이전트로 충분한지 먼저 비교하고, 멀티 에이전트가 품질·속도에서 이득일 때만 사용합니다.

**더 읽어보기 (영문):** [Multi-Agent Systems](https://outcomeschool.com/blog/multi-agent-systems)

</details>

<details>
<summary><b>Q23. 인과적 마스크를 적용한 스케일드 내적 어텐션(scaled dot-product attention)을 처음부터 구현하세요(NumPy 또는 PyTorch).</b></summary>

**NumPy 구현**

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)        # 수치 안정성 (오버플로 방지)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_attention(Q, K, V):
    """Q, K, V: (T, d_k). 반환: 출력 (T, d_v), 어텐션 가중치 (T, T)"""
    T, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)                 # (T, T) 스케일링
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # 대각선 위 = 미래 위치
    scores = np.where(mask, -np.inf, scores)        # 미래는 -inf → softmax 후 0
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

rng = np.random.default_rng(0)
T, d = 4, 8
Q, K, V = (rng.standard_normal((T, d)) for _ in range(3))
out, w = causal_attention(Q, K, V)
assert np.allclose(np.triu(w, 1), 0)               # 미래 가중치가 0인지 확인
assert np.allclose(w.sum(-1), 1)                   # 각 행의 합 = 1
```

**PyTorch 구현 (배치·헤드 차원 포함)**

```python
import math, torch

def sdpa_causal(q, k, v):
    """q, k, v: (B, H, T, d)"""
    T, d = q.size(-2), q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)          # (B, H, T, T)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v

q = k = v = torch.randn(2, 4, 16, 32)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(sdpa_causal(q, k, v), ref, atol=1e-5)  # 내장 구현과 비교 검증
```

**설명 포인트**

- `√d_k` 스케일링으로 점수 분산을 1 근처로 유지해 소프트맥스 포화를 막습니다.
- 마스크는 소프트맥스 **전에** 적용해야 합니다(후에 0으로 만들면 합이 1이 아님).
- 최댓값 빼기로 수치 안정성을 확보합니다.
- 실무에서는 Flash Attention 커널(`scaled_dot_product_attention`)을 사용합니다.

**더 읽어보기 (영문):** [Math behind Attention - Q, K, and V](https://outcomeschool.com/blog/math-behind-attention-qkv), [Causal Masking in Attention](https://outcomeschool.com/blog/causal-masking-in-attention)

</details>

<details>
<summary><b>Q24. 멀티 헤드 어텐션을 구현한 뒤, 이를 그룹 쿼리 어텐션으로 바꾸세요.</b></summary>

```python
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def split(self, x, n):                       # (B, T, n*dk) → (B, n, T, dk)
        B, T, _ = x.shape
        return x.view(B, T, n, self.dk).transpose(1, 2)

    def forward(self, x):
        q, k, v = self.split(self.wq(x), self.h), self.split(self.wk(x), self.h), self.split(self.wv(x), self.h)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (B, h, T, dk)
        B, _, T, _ = out.shape
        return self.wo(out.transpose(1, 2).reshape(B, T, self.h * self.dk))

class GroupedQueryAttention(MultiHeadAttention):
    """Q 헤드는 n_heads개, K/V 헤드는 n_kv_heads개. 각 K/V 헤드를 (n_heads/n_kv_heads)개 Q 헤드가 공유"""
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__(d_model, n_heads)
        assert n_heads % n_kv_heads == 0
        self.kv = n_kv_heads
        self.group = n_heads // n_kv_heads
        self.wk = nn.Linear(d_model, n_kv_heads * self.dk, bias=False)   # K/V 투영이 작아짐
        self.wv = nn.Linear(d_model, n_kv_heads * self.dk, bias=False)

    def forward(self, x):
        q = self.split(self.wq(x), self.h)                  # (B, h, T, dk)
        k = self.split(self.wk(x), self.kv)                 # (B, kv, T, dk)
        v = self.split(self.wv(x), self.kv)
        k = k.repeat_interleave(self.group, dim=1)          # (B, h, T, dk)로 확장해 공유
        v = v.repeat_interleave(self.group, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        B, _, T, _ = out.shape
        return self.wo(out.transpose(1, 2).reshape(B, T, self.h * self.dk))

x = torch.randn(2, 10, 512)
print(MultiHeadAttention(512, 8)(x).shape, GroupedQueryAttention(512, 8, 2)(x).shape)
```

**설명 포인트**

- 바뀌는 것은 **K·V 투영 크기와 헤드 공유** 뿐입니다. `n_kv_heads = n_heads`면 MHA, `n_kv_heads = 1`이면 MQA.
- **KV 캐시는 공유 전(n_kv_heads 기준)으로 저장** 하므로 캐시 크기가 `n_heads / n_kv_heads`배 줄어듭니다(예: 8 → 2면 1/4). 이것이 GQA의 핵심 이득입니다.
- `repeat_interleave`는 이해를 위한 구현이고, 최적화된 커널은 복사 없이 인덱스로 공유합니다.
- 기존 MHA 가중치를 GQA로 바꿀 때는 그룹 내 K·V 헤드 가중치를 평균 낸 뒤 소량 추가 학습(uptraining)합니다.

**더 읽어보기 (영문):** [Multi-Head Attention in Transformers](https://outcomeschool.com/blog/multi-head-attention-in-transformers), [Grouped Query Attention](https://outcomeschool.com/blog/grouped-query-attention)

</details>

<details>
<summary><b>Q25. 인과적 멀티 헤드 어텐션을 위한 KV 캐시와 단일 스텝 디코딩을 구현하세요.</b></summary>

```python
import math, torch
import torch.nn as nn

class CachedCausalMHA(nn.Module):
    def __init__(self, d_model, n_heads, max_len=4096):
        super().__init__()
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.max_len = max_len

    def init_cache(self, B, device, dtype):
        shape = (B, self.h, self.max_len, self.dk)
        return {"k": torch.zeros(shape, device=device, dtype=dtype),
                "v": torch.zeros(shape, device=device, dtype=dtype), "len": 0}

    def forward(self, x, cache):
        """x: (B, T_new, d). 프리필은 T_new = 프롬프트 길이, 디코드는 T_new = 1"""
        B, Tn, _ = x.shape
        q, k, v = self.qkv(x).split(self.h * self.dk, dim=-1)
        q, k, v = (t.view(B, Tn, self.h, self.dk).transpose(1, 2) for t in (q, k, v))

        start = cache["len"]
        end = start + Tn
        assert end <= self.max_len, "캐시 용량 초과"
        cache["k"][:, :, start:end] = k                  # 새 K/V를 캐시에 기록
        cache["v"][:, :, start:end] = v
        cache["len"] = end
        K, V = cache["k"][:, :, :end], cache["v"][:, :, :end]   # 과거 + 현재

        scores = q @ K.transpose(-2, -1) / math.sqrt(self.dk)   # (B, h, Tn, end)
        # 새 토큰 i(절대 위치 start+i)는 위치 0..start+i만 볼 수 있음
        q_pos = torch.arange(start, end, device=x.device)[:, None]
        k_pos = torch.arange(end, device=x.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ V                  # (B, h, Tn, dk)
        return self.wo(out.transpose(1, 2).reshape(B, Tn, -1))

# 검증: 캐시를 쓴 단계별 디코딩 결과 == 전체 시퀀스를 한 번에 계산한 결과
torch.manual_seed(0)
attn = CachedCausalMHA(64, 4).eval()
x = torch.randn(1, 6, 64)
with torch.no_grad():
    full = attn(x, attn.init_cache(1, x.device, x.dtype))        # 한 번에 6토큰
    cache = attn.init_cache(1, x.device, x.dtype)
    prefill = attn(x[:, :4], cache)                              # 프리필 4토큰
    steps = [attn(x[:, t:t + 1], cache) for t in (4, 5)]          # 디코드 1토큰씩
    stepwise = torch.cat([prefill, *steps], dim=1)
assert torch.allclose(full, stepwise, atol=1e-5)
```

**설명 포인트**

- **프리필:** 프롬프트 전체를 한 번에 넣어 캐시를 채움. **디코드:** 새 토큰 1개의 Q만 계산하고, K·V는 캐시에 추가 후 전체 캐시와 어텐션.
- 디코드 단계의 계산이 `O(T)`로 줄고(캐시 없이는 매번 `O(T²)` 재계산), 대신 캐시 메모리가 길이에 비례해 늘어납니다.
- 실제 엔진은 미리 할당한 큰 텐서 대신 PagedAttention 블록으로 관리하고, 배치마다 길이가 다른 경우를 처리합니다.

**더 읽어보기 (영문):** [What is KV Cache in LLMs?](https://outcomeschool.com/blog/kv-cache-in-llms)

</details>

<details>
<summary><b>Q26. BPE(Byte Pair Encoding)의 학습과 인코딩을 처음부터 구현하세요.</b></summary>

**바이트 수준 BPE** 구현입니다(모든 문자를 UTF-8 바이트 0~255로 시작해 미등록 문자가 없음).

```python
from collections import Counter

def get_pair_counts(seqs: dict[tuple, int]) -> Counter:
    counts = Counter()
    for seq, freq in seqs.items():
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += freq
    return counts

def merge_seq(seq: tuple, pair: tuple, new_id: int) -> tuple:
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(seq[i]); i += 1
    return tuple(out)

def train_bpe(texts: list[str], vocab_size: int):
    # 단어 단위로 미리 나누고(공백 포함) 빈도를 셈 → 단어 경계를 넘는 병합 방지
    words = Counter(w for t in texts for w in t.replace(" ", " Ġ").split(" ") if w)
    seqs = {tuple(w.encode("utf-8")): c for w, c in words.items()}
    merges: dict[tuple, int] = {}                          # (a, b) -> new_id, 학습 순서 유지
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    while next_id < vocab_size:
        counts = get_pair_counts(seqs)
        if not counts:
            break
        pair = max(counts, key=counts.get)                 # 가장 빈번한 쌍
        merges[pair] = next_id
        vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
        seqs = {merge_seq(s, pair, next_id): c for s, c in seqs.items()}
        next_id += 1
    return merges, vocab

def encode(text: str, merges: dict) -> list[int]:
    ids = list(text.encode("utf-8"))
    while len(ids) > 1:
        # 현재 시퀀스의 쌍 중 "가장 먼저 학습된" 병합을 적용 (학습 순서 = 우선순위)
        pairs = [(merges.get(p, float("inf")), i, p) for i, p in enumerate(zip(ids, ids[1:]))]
        rank, _, pair = min(pairs)
        if rank == float("inf"):
            break
        ids = list(merge_seq(tuple(ids), pair, merges[pair]))
    return ids

def decode(ids: list[int], vocab: dict) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

merges, vocab = train_bpe(["low lower lowest", "new newer newest"] * 50, vocab_size=280)
ids = encode("lowest newer", merges)
assert decode(ids, vocab) == "lowest newer"                # 손실 없는 복원
print(len("lowest newer".encode()), "바이트 →", len(ids), "토큰")
```

**설명 포인트**

- **학습:** 가장 빈번한 인접 쌍을 반복적으로 새 토큰으로 병합하고 병합 순서를 기록합니다.
- **인코딩:** 학습된 병합을 **학습된 순서(우선순위)대로** 적용해야 학습과 같은 분할이 나옵니다.
- 실제 토크나이저는 정규식 사전 분할(숫자·구두점·공백 규칙), 특수 토큰, 속도 최적화(우선순위 큐, Rust 구현)를 추가합니다. 위 예시의 공백 처리는 개념을 보여 주기 위한 단순화입니다.

**더 읽어보기 (영문):** [Byte Pair Encoding](https://outcomeschool.com/blog/bpe-in-llms)

</details>

<details>
<summary><b>Q27. 로짓 벡터에 대해 top-k, top-p, 온도 샘플링을 구현하세요.</b></summary>

```python
import numpy as np

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def sample(logits: np.ndarray, temperature=1.0, top_k: int | None = None,
           top_p: float | None = None, rng=np.random.default_rng()) -> int:
    logits = np.asarray(logits, dtype=np.float64).copy()

    # 1) 온도: 0이면 그리디
    if temperature <= 0:
        return int(np.argmax(logits))
    logits /= temperature

    # 2) top-k: 상위 k개 외에는 -inf
    if top_k is not None and 0 < top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf

    # 3) top-p (nucleus): 누적 확률이 p에 도달할 때까지의 최소 집합만 남김
    if top_p is not None and 0 < top_p < 1:
        probs = softmax(logits)
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep_n = int(np.searchsorted(cum, top_p) + 1)   # p를 처음 넘기는 토큰까지 포함
        remove = order[keep_n:]
        logits[remove] = -np.inf

    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))

logits = np.array([2.0, 1.5, 0.3, -1.0, -2.0])
print(sample(logits, temperature=0))                   # 그리디 → 0
print(sample(logits, temperature=0.7, top_k=3))
print(sample(logits, temperature=1.0, top_p=0.9))
```

**설명 포인트**

- **적용 순서:** 온도로 분포 모양을 조정한 뒤 top-k, top-p로 후보를 자르는 순서가 일반적입니다(라이브러리마다 순서가 조금 다를 수 있음).
- 같은 값을 가진 토큰이 여러 개면 top-k가 k개보다 많이 남을 수 있는데, 실무에서 큰 문제는 아닙니다.
- top-p는 최소 1개 토큰을 항상 남겨야 합니다(`keep_n ≥ 1`).
- 수치 안정성을 위해 소프트맥스 전 최댓값을 뺍니다.
- 배치 처리를 위해서는 같은 로직을 행렬 연산으로 벡터화합니다(PyTorch의 `torch.topk`, `torch.sort` + `cumsum`).

</details>

<details>
<summary><b>Q28. get/put이 O(1)인 LRU 캐시를 구현한 뒤, 항목별 TTL을 추가하세요.</b></summary>

**아이디어:** 해시맵(키 → 노드)으로 O(1) 조회, 이중 연결 리스트로 사용 순서를 O(1)에 갱신·제거합니다.

```python
import time

class Node:
    __slots__ = ("key", "value", "expires", "prev", "next")
    def __init__(self, key=None, value=None, expires=float("inf")):
        self.key, self.value, self.expires = key, value, expires
        self.prev = self.next = None

class LRUCacheTTL:
    def __init__(self, capacity: int, default_ttl: float | None = None, clock=time.monotonic):
        assert capacity > 0
        self.cap, self.default_ttl, self.clock = capacity, default_ttl, clock
        self.map: dict = {}
        self.head, self.tail = Node(), Node()            # 더미 노드: head 쪽이 가장 최근
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, n):
        n.prev.next, n.next.prev = n.next, n.prev

    def _push_front(self, n):
        n.prev, n.next = self.head, self.head.next
        self.head.next.prev = n
        self.head.next = n

    def _expired(self, n):
        return self.clock() >= n.expires

    def get(self, key, default=None):
        n = self.map.get(key)
        if n is None:
            return default
        if self._expired(n):                             # 만료 항목은 조회 시 제거 (지연 삭제)
            self._unlink(n); del self.map[key]
            return default
        self._unlink(n); self._push_front(n)             # 최근 사용으로 이동
        return n.value

    def put(self, key, value, ttl: float | None = None):
        ttl = ttl if ttl is not None else self.default_ttl
        expires = self.clock() + ttl if ttl is not None else float("inf")
        if (n := self.map.get(key)) is not None:
            n.value, n.expires = value, expires
            self._unlink(n); self._push_front(n)
            return
        if len(self.map) >= self.cap:
            lru = self.tail.prev                         # 가장 오래 사용 안 한 항목 제거
            self._unlink(lru); del self.map[lru.key]
        n = Node(key, value, expires)
        self.map[key] = n; self._push_front(n)

    def __len__(self):
        return len(self.map)

# 테스트 (가짜 시계로 TTL 검증)
t = [0.0]
c = LRUCacheTTL(2, clock=lambda: t[0])
c.put("a", 1, ttl=10); c.put("b", 2)
c.get("a"); c.put("c", 3)                                # b가 LRU로 제거됨
assert c.get("b") is None and c.get("a") == 1
t[0] = 11
assert c.get("a") is None                                # TTL 만료
```

**설명 포인트**

- get/put 모두 O(1). 만료는 **조회 시 지연 삭제** 로 처리해 O(1)을 유지합니다.
- 만료된 항목이 용량을 차지하는 문제를 줄이려면, 별도 최소 힙(만료 시각 기준)으로 주기적으로 정리하는 방식을 추가할 수 있습니다(정리 비용은 O(log n)).
- 멀티스레드 환경에서는 락이 필요합니다. 파이썬에서는 `collections.OrderedDict`(`move_to_end`, `popitem(last=False)`)로 더 짧게 구현할 수도 있습니다.

</details>

<details>
<summary><b>Q29. 토큰 수에 따라 비용이 늘어나는 LLM API용 토큰 버킷 속도 제한기를 구현한 뒤, 이를 분산 환경에서 동작하게 만드세요.</b></summary>

**1. 단일 프로세스 토큰 버킷 (비용 = LLM 토큰 수)**

```python
import time, threading

class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float, clock=time.monotonic):
        self.cap, self.rate, self.clock = capacity, refill_per_sec, clock
        self.tokens, self.last = capacity, clock()
        self.lock = threading.Lock()

    def _refill(self):
        now = self.clock()
        self.tokens = min(self.cap, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def try_acquire(self, cost: float) -> tuple[bool, float]:
        """성공 여부, 실패 시 기다려야 할 초"""
        with self.lock:
            self._refill()
            if cost > self.cap:
                return False, float("inf")               # 한 번에 감당 불가능한 요청
            if self.tokens >= cost:
                self.tokens -= cost
                return True, 0.0
            return False, (cost - self.tokens) / self.rate

    def refund(self, amount: float):                     # 예상보다 적게 쓴 토큰 반환
        with self.lock:
            self.tokens = min(self.cap, self.tokens + amount)

# 사용: 요청 전 예상 비용(입력 토큰 + max_tokens)으로 예약, 완료 후 실제 사용량으로 정산
bucket = TokenBucket(capacity=100_000, refill_per_sec=100_000 / 60)   # 분당 10만 토큰
est = count_tokens(prompt) + max_tokens
ok, wait = bucket.try_acquire(est)
if ok:
    usage = call_llm(...)
    bucket.refund(est - usage.total_tokens)
```

**2. 분산 버전 (Redis + Lua 스크립트로 원자적 처리)**

```python
LUA = """
local key = KEYS[1]
local cap = tonumber(ARGV[1]); local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3]); local cost = tonumber(ARGV[4])
local s = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(s[1]) or cap
local ts = tonumber(s[2]) or now
tokens = math.min(cap, tokens + math.max(0, now - ts) * rate)
local ok = 0; local wait = 0
if tokens >= cost then tokens = tokens - cost; ok = 1
else wait = (cost - tokens) / rate end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(cap / rate) * 2)
return {ok, tostring(wait)}
"""

class DistributedTokenBucket:
    def __init__(self, redis, name, capacity, refill_per_sec):
        self.r, self.key = redis, f"rl:{name}"
        self.cap, self.rate = capacity, refill_per_sec
        self.script = redis.register_script(LUA)

    def try_acquire(self, cost: float):
        now = self.r.time()                               # Redis 서버 시간 사용 (서버 간 시계 차이 방지)
        now = now[0] + now[1] / 1e6
        ok, wait = self.script(keys=[self.key], args=[self.cap, self.rate, now, cost])
        return bool(ok), float(wait)

# 키를 사용자·테넌트·모델별로 나눠 계층적 한도 적용: rl:user:123, rl:tenant:acme, rl:model:large
```

**설계 포인트**

- **원자성:** 읽기-계산-쓰기를 Lua 스크립트 하나로 실행해 여러 서버의 동시 요청에서도 정확합니다.
- **시계:** 각 서버 시계 대신 Redis 서버 시간 사용.
- **정산:** 완료 후 실제 토큰으로 환불(별도 `HINCRBYFLOAT` 스크립트).
- **장애 대비:** Redis 장애 시 로컬 버킷으로 보수적 폴백(허용 또는 차단 정책 결정).
- 거절 시 `429 + Retry-After(wait)` 반환.

</details>

<details>
<summary><b>Q30. 동시성 제한, 지터(jitter)를 적용한 재시도, 오류 격리를 갖추고 50,000개 문서에 LLM 호출을 실행하는 비동기 배치 처리기를 작성하세요.</b></summary>

```python
import asyncio, json, random, time
from pathlib import Path

class RetryableError(Exception): ...

async def call_with_retry(fn, doc, max_attempts=5, base=1.0, cap=60.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.wait_for(fn(doc), timeout=120)
        except (RetryableError, asyncio.TimeoutError):
            if attempt == max_attempts:
                raise
            await asyncio.sleep(random.uniform(0, min(cap, base * 2 ** attempt)))   # 풀 지터

async def run_batch(docs: list[dict], fn, out_path="results.jsonl", err_path="errors.jsonl",
                    concurrency=32):
    out, err = Path(out_path), Path(err_path)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [d for d in docs if d["id"] not in done]          # 재시작 시 완료분 건너뛰기 (재개 가능)

    sem = asyncio.Semaphore(concurrency)                     # 동시 실행 상한
    lock = asyncio.Lock()
    stats = {"ok": 0, "fail": 0, "start": time.monotonic()}

    async def write(path, record):
        async with lock:                                     # 파일 쓰기 직렬화
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def worker(doc):
        async with sem:
            try:
                result = await call_with_retry(fn, doc)
                await write(out, {"id": doc["id"], "result": result})
                stats["ok"] += 1
            except Exception as e:                           # 오류 격리: 한 문서 실패가 전체를 멈추지 않음
                await write(err, {"id": doc["id"], "error": repr(e)})
                stats["fail"] += 1
            n = stats["ok"] + stats["fail"]
            if n % 500 == 0:
                rate = n / (time.monotonic() - stats["start"])
                print(f"{n}/{len(todo)} 완료, 실패 {stats['fail']}, {rate:.1f}건/초")

    # 5만 개 태스크를 한꺼번에 만들지 않도록 청크 단위로 제출
    for i in range(0, len(todo), 2000):
        await asyncio.gather(*(worker(d) for d in todo[i:i + 2000]))
    return stats

async def summarize(doc):                                    # 실제 LLM 호출
    resp = await client.messages(model=MODEL, max_tokens=300,
                                 messages=[{"role": "user", "content": f"요약:\n{doc['text']}"}])
    if resp.status in (429, 500, 502, 503, 504):
        raise RetryableError(resp.status)
    return resp.text

# asyncio.run(run_batch(docs, summarize, concurrency=32))
```

**설계 포인트**

- **동시성 제한:** 세마포어로 제공사 속도 제한(RPM/TPM)에 맞춤. 토큰 기준 제한이 필요하면 Q29의 토큰 버킷과 결합.
- **지터 재시도:** 재시도할 오류만, 풀 지터로 재시도가 한 시점에 몰리지 않게.
- **오류 격리:** 실패를 별도 파일에 기록하고 계속 진행, 나중에 실패분만 재실행.
- **재개 가능성:** 결과를 즉시 append해 중단 후 재시작 시 완료분 건너뛰기.
- **메모리:** 청크 단위 제출.
- **비용 대안:** 실시간이 아니면 제공사 **배치 API**(할인, 비동기 처리)가 더 저렴하고 간단할 수 있습니다.

</details>

<details>
<summary><b>Q31. 청크 경계가 임의로 나뉘어도 처리할 수 있는, LLM 토큰 스트림용 SSE 파서를 작성하세요.</b></summary>

**어려운 점:** 네트워크 청크는 이벤트 경계와 무관하게 잘립니다. 한 이벤트가 여러 청크에 걸치거나, 한 청크에 여러 이벤트가 있거나, **한글 같은 멀티바이트 UTF-8 문자가 중간에서 잘릴 수** 있습니다. 줄바꿈도 `\n`, `\r\n`, `\r`이 섞일 수 있습니다.

```python
import codecs, json

class SSEParser:
    """바이트 청크를 받아 완성된 이벤트만 내보내는 증분 파서 (SSE 명세 기반)"""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()   # 잘린 UTF-8 바이트 보관
        self.buf = ""
        self._reset()

    def _reset(self):
        self.event, self.data, self.id = None, [], None

    def feed(self, chunk: bytes):
        self.buf += self.decoder.decode(chunk)
        while True:
            # 가장 먼저 나오는 줄바꿈 찾기 (\r\n, \n, \r 모두 지원)
            idx = min((i for i in (self.buf.find("\r\n"), self.buf.find("\n"), self.buf.find("\r"))
                       if i != -1), default=-1)
            if idx == -1:
                break                                            # 줄이 아직 완성되지 않음
            if self.buf.startswith("\r", idx) and idx == len(self.buf) - 1:
                break                                            # '\r' 뒤에 '\n'이 올 수도 있으니 대기
            line = self.buf[:idx]
            sep = 2 if self.buf.startswith("\r\n", idx) else 1
            self.buf = self.buf[idx + sep:]
            ev = self._line(line)
            if ev:
                yield ev

    def _line(self, line: str):
        if line == "":                                           # 빈 줄 = 이벤트 끝
            if not self.data:
                self._reset(); return None
            ev = {"event": self.event or "message", "data": "\n".join(self.data), "id": self.id}
            self._reset(); return ev
        if line.startswith(":"):
            return None                                          # 주석(keep-alive)
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self.data.append(value)                              # 여러 data 줄은 \n으로 합침
        elif field == "event":
            self.event = value
        elif field == "id":
            self.id = value
        return None

def stream_text(byte_chunks):
    p = SSEParser()
    for chunk in byte_chunks:
        for ev in p.feed(chunk):
            if ev["data"] == "[DONE]":
                return
            if ev["event"] == "error":
                raise RuntimeError(ev["data"])
            yield json.loads(ev["data"]).get("delta", "")

# 테스트: 한글 문자와 이벤트를 임의 위치에서 쪼개도 결과가 같아야 함
raw = 'data: {"delta": "안녕"}\n\ndata: {"delta": "하세요"}\r\n\r\ndata: [DONE]\n\n'.encode()
for size in (1, 2, 3, 7):
    chunks = [raw[i:i + size] for i in range(0, len(raw), size)]
    assert "".join(stream_text(chunks)) == "안녕하세요"
```

**핵심:** 증분 UTF-8 디코더, 줄 버퍼, 빈 줄을 이벤트 경계로 처리, 여러 `data:` 줄 결합, 주석·`[DONE]`·오류 이벤트 처리. 다양한 청크 크기로 같은 결과가 나오는지 테스트합니다.

**더 읽어보기 (영문):** [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)

</details>

<details>
<summary><b>Q32. 도구 디스패치, 오류 처리, 스텝 예산을 갖춘 최소한의 에이전트 루프를 구현하세요.</b></summary>

```python
import json, time
from dataclasses import dataclass, field

@dataclass
class Budget:
    max_steps: int = 10
    max_tokens: int = 50_000
    max_seconds: float = 120
    used_tokens: int = 0
    start: float = field(default_factory=time.monotonic)

    def exceeded(self, step: int) -> str | None:
        if step >= self.max_steps: return "단계 한도"
        if self.used_tokens >= self.max_tokens: return "토큰 한도"
        if time.monotonic() - self.start >= self.max_seconds: return "시간 한도"
        return None

def run_tool_loop(llm, messages, tools: dict, specs: list, budget: Budget | None = None):
    budget = budget or Budget()
    recent_calls = []
    for step in range(10_000):
        if reason := budget.exceeded(step):
            messages.append({"role": "user", "content":
                             f"{reason}에 도달했습니다. 도구를 더 쓰지 말고 지금까지의 결과와 남은 일을 요약하세요."})
            return llm(messages=messages, tools=[]).text, {"stopped": reason, "steps": step}

        resp = llm(messages=messages, tools=specs)
        budget.used_tokens += resp.usage.total_tokens
        messages.append({"role": "assistant", "content": resp.content})

        if not resp.tool_calls:                                   # 최종 답변
            return resp.text, {"stopped": "done", "steps": step + 1}

        for call in resp.tool_calls:
            sig = (call.name, json.dumps(call.arguments, sort_keys=True))
            recent_calls.append(sig)
            if recent_calls[-3:].count(sig) == 3:                 # 같은 호출 3회 반복 감지
                result = "같은 호출이 반복되고 있습니다. 다른 방법을 시도하거나 결론을 내리세요."
            elif call.name not in tools:
                result = f"오류: 알 수 없는 도구 '{call.name}'. 사용 가능: {list(tools)}"
            else:
                try:
                    out = tools[call.name](**call.arguments)
                    result = json.dumps(out, ensure_ascii=False, default=str)[:4000]   # 결과 길이 제한
                except TypeError as e:
                    result = f"인자 오류: {e}. 스키마를 확인하세요."
                except Exception as e:
                    result = f"실행 오류: {type(e).__name__}: {e}"   # 루프를 깨지 않고 관찰로 반환
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

**구성 요소 설명**

- **도구 디스패치:** 이름으로 함수를 찾아 실행, 없는 도구·잘못된 인자를 명확한 오류로 반환.
- **오류 처리:** 예외를 루프 밖으로 던지지 않고 모델에 관찰로 돌려줘 스스로 복구하게 함.
- **예산:** 단계·토큰·시간 한도, 도달 시 그냥 끊지 않고 요약을 요청해 우아하게 종료.
- **반복 감지:** 같은 도구·인자 반복을 감지해 전략 변경 유도.
- **결과 길이 제한:** 컨텍스트 폭증 방지.

**실무 확장:** 위험 도구의 사람 승인, 도구 병렬 실행, 컨텍스트 압축, 트레이싱(단계별 스팬), 체크포인트로 재개.

**더 읽어보기 (영문):** [AI Agent Loop](https://outcomeschool.com/blog/ai-agent-loop)

</details>
