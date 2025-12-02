[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/r1tAQ0HC)
# Literature Review Assistant - Multi-Agent System

A multi-agent literature review system powered by LangGraph, Groq API, and Semantic Scholar. This system helps researchers conduct comprehensive literature reviews by orchestrating specialized agents that search, analyze, and synthesize academic papers.

## 🌟 Features

- **4 Specialized Agents** orchestrated via LangGraph:
  - 🎯 **Planner**: Analyzes research topics and creates search strategies
  - 🔍 **Researcher**: Searches academic papers via Semantic Scholar API
  - 🔬 **Analyzer**: Identifies patterns, methodologies, and state-of-the-art
  - ✍️ **Writer**: Synthesizes findings into comprehensive literature reviews

- **Safety Guardrails**: Integrated Guardrails AI for input/output validation
- **Interactive Web UI**: Streamlit interface with real-time agent traces
- **LLM-as-a-Judge Evaluation**: Comprehensive evaluation with 6 criteria
- **Citation Management**: Automatic APA-style citation generation

## 🎬 Demo

### System UI Screenshot
The web interface provides an intuitive experience with:
- **Gradient hero header** with system branding
- **Real-time agent activity traces** showing each agent's actions
- **Tavily web search indicator** (green badge when used)
- **LLM Judge evaluation scores** with detailed breakdown
- **Safety event notifications** when content is flagged

![Streamlit UI](artifacts/ui_screenshot.png)
*Main interface showing query input, agent traces, and evaluation results*

### Sample Outputs

**📄 Full Session Export**: [`artifacts/sample_session.json`](artifacts/sample_session.json)  
Complete JSON export of an end-to-end session including all agent traces, papers found, and judge evaluation.

**📝 Literature Review**: [`artifacts/sample_review.md`](artifacts/sample_review.md)  
Sample generated literature review on "Design patterns for accessible user interfaces in mobile applications" with 8 papers analyzed.

**⚖️ Judge Evaluation**: [`artifacts/judge_evaluation_results.json`](artifacts/judge_evaluation_results.json)  
LLM-as-a-Judge results from evaluating 5 diverse queries with detailed scoring breakdown.

## 📁 Project Structure

```
.
├── src/
│   ├── agents/
│   │   └── langgraph_agents.py    # 4 LangGraph agents
│   ├── langgraph_orchestrator.py  # Agent orchestration with LangGraph
│   ├── guardrails/
│   │   └── safety_manager.py      # Safety guardrails
│   ├── tools/
│   │   ├── paper_search.py        # Semantic Scholar integration
│   │   ├── web_search.py          # Tavily/Brave search
│   │   └── citation_tool.py       # APA citation formatting
│   ├── evaluation/
│   │   └── judge.py               # LLM-as-a-Judge implementation
│   └── ui/
│       └── streamlit_app.py       # Web interface
├── data/
│   └── example_queries.json       # Test queries for evaluation
├── config.yaml                    # System configuration
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment variables template
└── main.py                       # Main entry point
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`
- Virtual environment

### 2. Installation

#### Installing uv (Recommended)

`uv` is a fast Python package installer and resolver. Install it first:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Using pip
pip install uv
```

#### Setting up the Project

Clone the repository and navigate to the project directory:

```bash
cd is-492-assignment-3
```

**Option A: Using uv (Recommended - Much Faster)**

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

**Option B: Using standard pip**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # On macOS/Linux
# OR
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Security Setup (Important!)

**Before committing any code**, set up pre-commit hooks to prevent API key leaks:

```bash
# Quick setup - installs hooks and runs security checks
./scripts/install-hooks.sh

# Or manually
pre-commit install
```

This will automatically scan for hardcoded API keys and secrets before each commit. See `SECURITY_SETUP.md` for full details.

### 4. API Keys Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required: At least one LLM API
GROQ_API_KEY=your_groq_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# Recommended: At least one search API
TAVILY_API_KEY=your_tavily_api_key_here
# OR
BRAVE_API_KEY=your_brave_api_key_here

# Optional: For academic paper search
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

#### Getting API Keys

- **Groq** (Recommended for students): [https://console.groq.com](https://console.groq.com) - Free tier available
- **OpenAI**: [https://platform.openai.com](https://platform.openai.com) - Paid, requires credits
- **Tavily**: [https://www.tavily.com](https://www.tavily.com) - Student free quota available
- **Brave Search**: [https://brave.com/search/api](https://brave.com/search/api)
- **Semantic Scholar**: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) - Free tier available

### 5. Configuration

Edit `config.yaml` to customize your system:

- Choose your research topic
- **Configure agent prompts** (see below)
- Set model preferences (Groq vs OpenAI)
- Define safety policies
- Configure evaluation criteria
### 5. Configuration

The system is pre-configured for literature reviews in `config.yaml`. Key settings:

- **System**: Max iterations, timeout settings
- **Agents**: Planner, Researcher, Analyzer, Writer configurations
- **Models**: Groq API with llama-3.1-70b-versatile
- **Tools**: Semantic Scholar and Tavily search
- **Safety**: Guardrails policies and prohibited categories
- **Evaluation**: 6 criteria for LLM-as-a-Judge

You can customize agent behaviors by editing `config.yaml` if needed.

---

## 🚀 Running the System

### Quick Start (Single Command Demo)

To run a complete end-to-end demonstration with all agents:

```bash
python main.py --mode test
```

**This will:**
1. ✅ Run safety checks on the input query
2. 📋 Create a search strategy (Planner agent)
3. 🌐 Search with Tavily for supplementary sources
4. 🔍 Find academic papers via SerpAPI/Google Scholar (Researcher agent)
5. 🔬 Analyze patterns and themes (Analyzer agent)
6. ✍️ Draft a literature review (Writer agent)
7. ✅ Quality check the draft
8. 🛡️ Safety check the output
9. ⚖️ Evaluate with LLM-as-a-Judge

**Expected Output:**
- Complete literature review (6000+ characters)
- 8-10 academic papers with citations
- Agent execution traces showing all steps
- Judge evaluation score (typically 7-9/10)
- Duration: ~20-30 seconds

**Sample Output Location:** `outputs/test_run_sample.log`

---

### Web Interface (Recommended)

Launch the interactive Streamlit UI:

```bash
python main.py --mode web
# OR
streamlit run src/ui/streamlit_app.py
```

Then open your browser to **http://localhost:8501**

**Features:**
- 🎨 Beautiful gradient UI with modern design
- 📊 Real-time agent activity traces
- 🌐 Tavily usage indicator (green badge)
- ⚖️ LLM Judge scores with detailed breakdown
- 🛡️ Safety event notifications
- 💾 Session history

**Try these example queries:**
1. "Design patterns for accessible user interfaces"
2. "Ethical considerations in AI-driven education"
3. "Usability challenges in augmented reality interfaces"

---

### Web Interface (Recommended)

Start the Streamlit web interface:

```bash
python main.py --mode web
# OR directly:
streamlit run src/ui/streamlit_app.py
```

The interface provides:
- Input forms for research topic and project description
- Real-time agent activity traces
- Formatted literature review output
- Interactive citation links
- Safety event notifications
- Query history

### Test Mode

Run a quick test with a sample query:

```bash
python main.py --mode test
```

This will process a sample literature review query and display results in the terminal.

### Evaluation Mode

Run comprehensive evaluation on test queries:

```bash
python main.py --mode evaluate
```

This will:
- Load test queries from `data/example_queries.json`
- Process each query through the multi-agent system
- Evaluate outputs using LLM-as-a-Judge with 6 criteria
- Generate detailed evaluation report in `outputs/evaluations/`
- Display aggregate scores and statistics

---

## 📊 Evaluation Criteria

The LLM-as-a-Judge evaluates literature reviews on 6 criteria:

1. **Relevance & Coverage** (20%): Appropriate scope, relevant papers included
2. **Evidence Quality** (20%): Authoritative sources, proper APA citations
3. **Comparative Analysis** (20%): Pattern identification, approach comparison
4. **Factual Accuracy** (15%): Correct paper details, no hallucinations
5. **Safety Compliance** (10%): Professional tone, no inappropriate content
6. **Clarity & Organization** (15%): Logical structure, clear writing

Scores range from 0-10 for each criterion, with weighted aggregation for overall score.

---

## 🛡️ Safety Guardrails

The system implements multiple safety layers:

**Prohibited Categories:**
- Harmful research topics (weapons, illegal activities)
- Personal attacks or biased language
- Academic dishonesty (plagiarism requests)
- Off-topic or inappropriate queries

**Response Strategies:**
- **Input Validation**: Blocks unsafe queries before processing
- **Output Sanitization**: Filters or refuses inappropriate responses
- **Event Logging**: Records all safety events with context

Safety events are displayed in the UI and logged to `logs/safety_events.log`.

---

## 🔧 System Architecture

```
User Query → Safety Check (Input) → Planner Agent
                                        ↓
                                   Researcher Agent
                                        ↓ 
                                   Analyzer Agent
                                        ↓
                                   Writer Agent
                                        ↓
                                  Quality Check
                                   ↙        ↘
                              Approve    Revise (max 2x)
                                 ↓
                            Safety Check (Output)
                                 ↓
                            Final Review + Citations
```

**LangGraph Workflow:**
- State-based orchestration with typed state schema
- Conditional edges for revision loops
- Error handling at each node
- Transparent agent traces for debugging

---

## 📝 Example Usage

**Research Topic:**
```
Design patterns for accessible user interfaces in mobile applications
```

**Project Description:**
```
I'm researching how to make mobile apps more accessible for users with 
disabilities, particularly focusing on screen reader support, touch target 
sizes, and color contrast. I need to understand current best practices and 
emerging techniques.
```

**Generated Output:**
The system will:
1. Create a search strategy targeting accessibility, mobile UI, design patterns
2. Search Semantic Scholar for relevant papers (2018-2024)
3. Analyze papers to identify common themes and patterns
4. Synthesize a comprehensive literature review with proper APA citations
5. Include bibliography with 8-15 authoritative sources

---

## 📚 Research Tools

### Semantic Scholar Integration
- **Source**: Academic papers database
- **Coverage**: Computer Science, HCI, AI/ML
- **Features**: Citation counts, abstracts, author info, PDF links
- **Filters**: Year range, citation threshold

### Web Search (Optional)
- **Tavily**: AI-optimized search with deep crawling
- **Brave**: Privacy-focused search alternative
- **Use Case**: Supplementary sources, recent developments

---

##

## Testing

Run tests (if you create them):

```bash
pytest tests/
```

## Resources

### Documentation
- [uv Documentation](https://docs.astral.sh/uv/) - Fast Python package installer
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Guardrails AI](https://docs.guardrailsai.com/)
- [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/)
- [Tavily API](https://docs.tavily.com/)
- [Semantic Scholar API](https://api.semanticscholar.org/)
