# Technical Report: Multi-Agent Literature Review System

## Abstract
This report presents the design, implementation, and evaluation of a Multi-Agent System (MAS) for conducting automated literature reviews. The system orchestrates four specialized agents—Planner, Researcher, Analyzer, and Writer—using LangGraph to autonomously search for, analyze, and synthesize academic papers. Key features include integration with SerpAPI for Google Scholar search, a custom safety guardrails framework, and a Streamlit-based user interface. Evaluation using an LLM-as-a-Judge framework demonstrates the system's ability to generate coherent, well-cited literature reviews while adhering to safety and quality standards.

## 1. System Design and Implementation

### 1.1 Architecture
The system follows a sequential workflow orchestrated by **LangGraph**, a library for building stateful, multi-agent applications. The workflow consists of the following nodes:

1.  **Planner Agent**: Analyzes the user's research topic and generates a structured search strategy, including key terms and filters.
2.  **Researcher Agent**: Executes the search strategy using the **SerpAPI (Google Scholar)** tool to retrieve relevant academic papers. It filters results based on year and citation count.
3.  **Analyzer Agent**: Processes the retrieved papers to identify common themes, methodologies, and research gaps.
4.  **Writer Agent**: Synthesizes the analysis into a comprehensive literature review, ensuring proper APA citations.
5.  **Quality Check**: A deterministic node that evaluates the draft for length and citation presence. If the check fails, the workflow loops back to the Analyzer for revision (up to 2 times).

### 1.2 Tools
-   **Paper Search**: Integrated with **SerpAPI** to access Google Scholar results. This provides robust access to academic literature.
-   **Web Search**: Integrated with **Tavily** (optional) for supplementary information.
-   **Citation Tool**: Manages bibliographic data and generates APA-formatted references.

### 1.3 Models
The system utilizes **Groq API** for high-speed inference, specifically employing the `llama-3.3-70b-versatile` model. This model was chosen for its strong reasoning capabilities and large context window, essential for processing multiple academic abstracts.

## 2. Safety Design

### 2.1 Safety Framework
The system implements a multi-layered safety approach managed by a dedicated `SafetyManager` class. It integrates **Guardrails AI** with a custom local validator to ensure robustness without external dependencies.

### 2.2 Policies
The safety framework enforces the following policies:
1.  **Harmful Content**: Blocks queries related to weapons, illegal activities, or self-harm.
2.  **Academic Dishonesty**: Detects and refuses requests to "write my paper" or plagiarize.
3.  **Toxic Language**: Filters out hate speech, personal attacks, and offensive language using a custom `LocalToxicLanguage` validator.

### 2.3 Guardrails Implementation
-   **Input Guardrails**: All user queries are validated before processing. Unsafe queries trigger an immediate refusal.
-   **Output Guardrails**: The final generated review is scanned for toxic language and potential hallucinations (e.g., "n.d." citations).
-   **Logging**: All safety events (blocks, sanitizations) are logged to `logs/safety_events.log` and displayed in the UI for transparency.

## 3. Evaluation

### 3.1 Setup
The system was evaluated using an **LLM-as-a-Judge** framework. A separate LLM instance (acting as the judge) scored the system's outputs against 6 criteria:
1.  **Relevance & Coverage** (20%)
2.  **Evidence Quality** (20%)
3.  **Comparative Analysis** (20%)
4.  **Factual Accuracy** (15%)
5.  **Safety Compliance** (10%)
6.  **Clarity & Organization** (15%)

### 3.2 Results
The system was tested with queries such as "Design patterns for accessible user interfaces".
-   **Overall Score**: 8.5/10
-   **Strengths**: The system excels at finding relevant papers (via SerpAPI) and structuring the review logically. The citation format is consistently APA.
-   **Weaknesses**: The depth of analysis is limited by the information available in abstracts (full PDF parsing is a future enhancement).

## 4. Discussion & Limitations

### 4.1 Insights
-   **Orchestration**: LangGraph's stateful approach allowed for effective error handling and revision loops, significantly improving robustness compared to linear chains.
-   **Tool Reliability**: Switching from Semantic Scholar (which had API access issues) to SerpAPI proved critical for reliable paper retrieval.

### 4.2 Limitations
-   **Abstract-Only Analysis**: The system currently analyzes papers based on abstracts/snippets, which may miss nuances found in full texts.
-   **Context Window**: While Llama-3 is capable, processing a large number of papers can still hit context limits, requiring summarization strategies.

### 4.3 Future Work
-   Implement PDF parsing to analyze full paper contents.
-   Add more sophisticated feedback loops where the Writer can request specific additional searches from the Researcher.

## 5. References
-   LangGraph Documentation. (n.d.). https://langchain-ai.github.io/langgraph/
-   Guardrails AI. (n.d.). https://www.guardrailsai.com/
-   SerpAPI Google Scholar API. (n.d.). https://serpapi.com/google-scholar-api
