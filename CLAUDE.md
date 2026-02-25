# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
uv sync
```

**Run the application:**
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Add a dependency:**
```bash
uv add <package>
```

There are no tests or linter configurations in this project.

## Environment

Requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

The server loads documents from `../docs` (relative to `backend/`) on startup and persists vectors to `backend/chroma_db/`. ChromaDB is skipped for courses already present — delete `backend/chroma_db/` to force a full reload.

## Architecture

The app is a RAG chatbot that answers questions about course materials. FastAPI serves both the REST API and the static frontend from a single process.

**Request flow for a user query:**
1. `frontend/script.js` POSTs `{ query, session_id }` to `/api/query`
2. `backend/app.py` creates a session if needed, delegates to `RAGSystem.query()`
3. `rag_system.py` fetches conversation history and calls `AIGenerator.generate_response()` with tool definitions
4. `ai_generator.py` makes a first Claude API call; if Claude requests a tool, `_handle_tool_execution()` runs it and makes a second call to synthesise the final answer
5. Tool execution hits `CourseSearchTool` → `VectorStore.search()` → ChromaDB semantic search
6. Sources from the search are returned alongside the answer and rendered in the frontend

**Document ingestion flow (startup):**
`app.py` → `RAGSystem.add_course_folder()` → `DocumentProcessor.process_course_document()` → `VectorStore.add_course_metadata()` + `add_course_content()`

**Two ChromaDB collections:**
- `course_catalog` — one record per course (title, instructor, link, lessons as JSON string); used for fuzzy course-name resolution
- `course_content` — one record per text chunk; used for semantic search at query time

**Tool calling pattern:**
`search_tools.py` defines tools in Anthropic's tool schema format. `ToolManager` registers them and is passed to `AIGenerator`, which forwards definitions to Claude and routes `tool_use` responses back through `ToolManager.execute_tool()`. To add a new tool, subclass `Tool`, implement `get_tool_definition()` and `execute()`, and register it with `ToolManager` in `rag_system.py`.

**Session history:**
`SessionManager` stores conversation in memory (resets on server restart). History is serialised as a plain string and prepended to the system prompt — it is not sent as separate message turns. `MAX_HISTORY = 2` means the last 2 exchanges (4 messages) are retained per session.

**Course document format:**
Documents in `docs/` must follow a specific header format (parsed by `document_processor.py`):
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<content...>
```
Chunks are sentence-aware, 800 chars with 100-char overlap. The first chunk of each lesson is prefixed with `"Lesson N content: "`.

**Amazon Bedrock alternative:**
`backend/ai_generator_aws.py` is a drop-in replacement for `ai_generator.py` that uses `boto3` and the Bedrock Converse API instead of the Anthropic SDK. Swap the import in `rag_system.py` and change the constructor call to `AIGenerator(model="us.anthropic.claude-sonnet-4-20250514-v1:0", region="us-east-1")` — no API key needed (uses AWS credential chain).
