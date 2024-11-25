# Document Classification

In document intelligence, classification is often the crucial first step. It sets the stage for subsequent processes like data extraction and analysis. Before the rise of LLMs, this used to be accomplished (and still is) with AI models training in-house for certain use cases. Services such as Azure Document Intelligence give you this feature, but they are not dynamic and will set you up for "Vendor lock-in".

LLMs may not be the most efficient for this task, but they are agnostic and near-perfect for it.

<div align="center">
  <img src="../../../assets/classification_overview.png" alt="Classification Overview">
</div>

## Classification Techniques

<div class="grid cards">
    <ul>
        <li>
            <p><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2m0 16H5V5h14v14m-2-2H7v-2h10v2m-10-4h10v2H7v-2m10-6v2H7V7h10Z"></path></svg></span> <strong>Basic Classification</strong></p>
            <p>Simple yet powerful classification using a single LLM with contract mapping.</p>
            <p><a href="basic"><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8.22 2.97a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.042-.018.75.75 0 0 1-.018-1.042l2.97-2.97H3.75a.75.75 0 0 1 0-1.5h7.44L8.22 4.03a.75.75 0 0 1 0-1.06"></path></svg></span> Learn More</a></p>
        </li>
        <li>
            <p><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M16 17v2H2v-2s0-4 7-4 7 4 7 4m-7-6a4 4 0 0 0 4-4 4 4 0 0 0-4-4 4 4 0 0 0-4 4 4 4 0 0 0 4 4m8.8 4c1.2.7 2.2 1.7 2.2 3v2h3v-2s0-2.9-5.2-3M15 4a4 4 0 0 0 1.8 3.3A4 4 0 0 1 19 11c1.9 0 3-1.3 3-3a4 4 0 0 0-4-4h-3Z"></path></svg></span> <strong>Mixture of Models (MoM)</strong></p>
            <p>Enhance accuracy by combining multiple models with different strategies.</p>
            <p><a href="mom"><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8.22 2.97a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.042-.018.75.75 0 0 1-.018-1.042l2.97-2.97H3.75a.75.75 0 0 1 0-1.5h7.44L8.22 4.03a.75.75 0 0 1 0-1.06"></path></svg></span> Learn More</a></p>
        </li>
        <li>
            <p><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M3 3h18v2H3V3m0 16h18v2H3v-2m0-8h18v2H3v-2m0 4h8v2H3v-2m0-8h8v2H3V7m8 4h10v2H11v-2m0 8h10v2H11v-2m0-8h10v2H11V7"></path></svg></span> <strong>Tree-Based Classification</strong></p>
            <p>Handle complex hierarchies and similar document types efficiently.</p>
            <p><a href="tree"><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8.22 2.97a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.042-.018.75.75 0 0 1-.018-1.042l2.97-2.97H3.75a.75.75 0 0 1 0-1.5h7.44L8.22 4.03a.75.75 0 0 1 0-1.06"></path></svg></span> Learn More</a></p>
        </li>
        <li>
            <p><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 9a3 3 0 0 0-3 3 3 3 0 0 0 3 3 3 3 0 0 0 3-3 3 3 0 0 0-3-3m0 8a5 5 0 0 1-5-5 5 5 0 0 1 5-5 5 5 0 0 1 5 5 5 5 0 0 1-5 5m0-12.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5Z"></path></svg></span> <strong>Vision Classification</strong></p>
            <p>Leverage visual features for better accuracy.</p>
            <p><a href="vision"><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8.22 2.97a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.042-.018.75.75 0 0 1-.018-1.042l2.97-2.97H3.75a.75.75 0 0 1 0-1.5h7.44L8.22 4.03a.75.75 0 0 1 0-1.06"></path></svg></span> Learn More</a></p>
        </li>
    </ul>
</div>

## Classification Response

All classification methods return a standardized response:

```python
from typing import Optional
from pydantic import BaseModel, Field

class ClassificationResponse(BaseModel):
    name: str
    confidence: Optional[int] = Field(
        description="From 1 to 10. 10 being the highest confidence",
        ge=1, 
        le=10
    )
```

## Available Strategies

ExtractThinker supports three main classification strategies:

- **CONSENSUS**: All models must agree on the classification
- **HIGHER_ORDER**: Uses the result with highest confidence
- **CONSENSUS_WITH_THRESHOLD**: Requires consensus and minimum confidence

## Common Challenges

1. **Large Context Windows**: More classifications mean larger contexts
2. **Similar Documents**: Distinguishing between similar document types
3. **Confidence Levels**: Ensuring high confidence in classifications
4. **Scalability**: Handling growing number of document types

## Best Practices

- Start with basic classification for simple cases
- Use MoM for critical classifications
- Implement tree-based approach for similar documents
- Consider vision classification for complex layouts
- Set appropriate confidence thresholds
- Monitor and log classification results

For detailed implementation of each technique, visit their respective pages. 