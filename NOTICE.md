# Third-Party Notices

The main project license is MIT (see `LICENSE`). A few small generic snippets
come from MIT-licensed upstream projects and keep their original copyright
notices below. No upstream source trees are vendored in this repository.

## Included snippets (MIT)

### LightRAG — MIT License

Source: https://github.com/HKUDS/LightRAG

The `EmbeddingFunc` dataclass and `wrap_embedding_func_with_attrs` helper in
`src/rag_comparison/backends/graph/hirag.py` and
`src/rag_comparison/backends/graph/lightrag.py` were adapted from
`lightrag/utils.py` in LightRAG. The upstream license is:

```
MIT License

Copyright (c) 2025 LightRAG Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### HiRAG — MIT License

Source: https://github.com/HKUDS/HiRAG

HiRAG is MIT-licensed with `Copyright (c) 2025 haoyuhuang`. Its package is not
redistributed here; only the two small generic helper snippets described above
are retained. The same MIT terms apply, with that copyright line.

## Referenced but not redistributed (Apache-2.0)

### FastChat (arena UI) — Apache License 2.0

Source: https://github.com/lm-sys/FastChat

FastChat is an upstream arena/Gradio project licensed under the Apache License,
Version 2.0. This repository does not include FastChat source. `ui/app.py` is
an independent lightweight replacement that calls this project's own
OpenAI-style API.

Full license text: https://www.apache.org/licenses/LICENSE-2.0 and
https://github.com/lm-sys/FastChat/blob/main/LICENSE. Because no FastChat code
is included, its NOTICE obligations do not attach to this repository; this
entry is provided for attribution only.

## Other referenced projects

- LangChain (MIT), FastAPI (MIT), Gradio (Apache-2.0), Requests (Apache-2.0),
  Pydantic (MIT), Milvus (Apache-2.0), Chroma (Apache-2.0), pgvector (PostgreSQL
  License), Neo4j (GPL/AGPL/commercial editions) — used as dependencies or
  optional adapters; see each project's own license.
- HippoRAG — https://github.com/OSU-NLP-Group/HippoRAG — referenced as an
  optional external backend; not vendored here. Consult its upstream license
  before use.

## Why the main project uses MIT

The authored code and the retained MIT snippets can be released under MIT with
this notice. No Apache-2.0 code is redistributed here, so Apache-2.0 is not
required for the authored code.
