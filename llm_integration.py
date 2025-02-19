from llama_cpp import Llama

# Load Mistral model
llm = Llama(
    model_path="llm_models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    n_ctx=8192,
    n_threads=6,
    use_mmap=True,
    use_mlock=True
)

def clean_text(text):
    """Ensure UTF-8 encoding and remove unnecessary characters."""
    return text.encode("utf-8", "ignore").decode("utf-8").strip()

def truncate_text(text, max_tokens=2048):
    """Ensure the text does not exceed the model’s context limit."""
    words = text.split()
    return " ".join(words[:max_tokens])

def generate_initial_prompt(document_text, analysis_data=None):
    """Generate the initial assessment prompt."""
    document_text = clean_text(truncate_text(document_text))

    # Include metrics if available
    if analysis_data:
        metrics_section = f"""
### Document Metrics:
- Complexity: {analysis_data['complexity']}
- Vocabulary Diversity: {analysis_data['vocabulary_diversity']:.2f}
- Writing Style: {'Passive' if analysis_data['style'] > 0 else 'Active'}
- Sentiment: Negative={analysis_data['details']['sentiment']['neg']:.2f}, Neutral={analysis_data['details']['sentiment']['neu']:.2f}, Positive={analysis_data['details']['sentiment']['pos']:.2f}
- Key Terms: {', '.join(analysis_data['key_terms'])}
- Main Topics: {', '.join(analysis_data['topics'])}
"""
    else:
        metrics_section = ""

    prompt = f"""
You are a cybersecurity expert analyzing a document. Your task is to evaluate its cybersecurity approach, risks, and compliance with best practices.

### Document Excerpt:
{document_text}

{metrics_section}

### Required Analysis:
1. **Risk Identification**: Identify cybersecurity risks in the document.
2. **Threat Mitigation**: Assess if the document proposes clear, effective security strategies.
3. **Compliance & Best Practices**: Compare the document’s approach to standards like NIST, ISO 27001.
4. **Recommendations**: Suggest practical and actionable improvements.

### Expected Response:
1. Summary
2. Identified Risks
3. Suggested Improvements
"""
    return prompt

def generate_refined_prompt(initial_response, document_text):
    """Generate the refined analysis prompt."""
    prompt = f"""
You have already provided an initial cybersecurity assessment of the document. Now, refine your analysis to improve its clarity, depth, and alignment with industry best practices.

### Initial Assessment:
{initial_response}

### Additional Considerations:
- Expand on any vague or generic risks.
- Provide specific solutions for unclear recommendations.
- Compare the document more explicitly to NIST, ISO 27001.
- Ensure the response is actionable and precise.

### Expected Final Analysis:
1. Strengths & Weaknesses
2. Clearer Cybersecurity Risks
3. Stronger Compliance Evaluation
4. Refined Recommendations
"""
    return prompt

def analyze_with_llm(document_text, analysis_data=None):
    """Process the document using Mistral with optional metrics."""
    # Generate initial prompt
    initial_prompt = generate_initial_prompt(document_text, analysis_data)
    
    try:
        # First round: Initial assessment
        initial_response = llm(initial_prompt, max_tokens=1000, temperature=0.3, top_p=0.85)
        initial_response = initial_response["choices"][0]["text"].strip()

        # Second round: Refined analysis
        refined_prompt = generate_refined_prompt(initial_response, document_text)
        refined_response = llm(refined_prompt, max_tokens=1000, temperature=0.3, top_p=0.85)
        refined_response = refined_response["choices"][0]["text"].strip()

        return refined_response
    except Exception as e:
        return f"Error in LLM response: {str(e)}"