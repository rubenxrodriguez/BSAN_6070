# 1.1 Key Innovations
  * LSTM allows for longer context, but when the sentence is long, the context vector (memory cell) becomes complex. Need for attention to specific parts and avoid "noise"
  *  Self-Attention:
      - Weights input relationships dynamically (e.g., pronoun resolution)
      - Processes all positions in parallel (vs RNN sequential)

# 2) BERT (Bi Encoder Representations from Transformers)
  * LSTM allows for longer context, but when the sentence is long, the context vector (memory cell) becomes complex. Need for attention to specific parts and avoid "noise"
  * BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
  * Architecture - multi-layer bidirectional transformer encoder
  * Consists of pre-training and fine-tuning
## 2.1 Sequence
  * A sequence refers to the input token sequence to BERT, which can be a single or two sentences packed together. Sentences are first "tokenized"
## 2.2 Pre-Training Tasks
  * Unsupervised Learning used for BERT pre-training
  * Masked Language Modeling (MLM)
      - 15% of input tokens are masked randomly and then predicted
      - "The capital of [MASK] is Paris" -> "France"
  * Next Sentence Prediction (NSP)
      - Binary Classification of sentence pairs - predict if two sentences are related
## 2.3 Fine-Tuning
  * Fine-tuning is adding a layer of untrained neurons as a feedforward layer on top of the pre-trained BERT.
  * Pre-trained language model is tweaked through backpropagation
  * Pre-Training is expensive, fine-tuning is fast and inexpensive
## 2.4 Applications
  * A pre-trained BERT can be fine-tuned to solve multiple NLP tasks like Text Summarization, Sentiment Analysis, Q&A chatbots, Translation

# 3. Transformers
## 3.1 Attention
  * basic idea is to avoid attempting to learn a single vector representation for each sentence, instead, pay attention to specific input vectors based on attention weights
  * BERT is good at translating, transformers give you speed, accuracy, and understanding long term dependency.
  * Parallelization
  * gained knowledge can now be fine-tuned to a variety of language tasks (classifying, translating, sentiment, NSP)


# 4. Generative AI Models
  * Large Language Models (LLMs) - GPT/LLaMa/PaLM/BLOOM that are built on transformers architecture
      - GPT/Llama predict tokens autoregressively (Decoder-only)
      - PaLM - use an encoder and decoder (like original Transformer)
## 4.1 Text Generation
  * Autoregressive Process:
      - Predicts next token given previous ones # sometimes random sometimes probabilistic ??
      - Iteratively builds output (left-to-right by default)
  * Prompting Strategies:
      - Zero-shot: Direct task instruction ("Classify this tweet:")
      - Few-shot: Provide examples in prompt
      - Chain-of-thought: Explicit reasoning steps
## 4.2 BERT vs GPT (pretrained transformer)
  * Bert
      - bidirectional
      - good at sentence completion
      - sentiment analysis/ Q&A/Named Entity Recognition
      - pretrained on unlabeled data from BooksCorpus and Wikipedia
  * ChatGPT
      - uni-directional / autoregressive
      - good at generating coherent text from a given prompt
      - chatbots/ creative writing / summarization
      - pretrained from half-billion tokens from books/websites/etc
## 4.3 Hyperparameters
  * Temperature (controls randomness [low = deterministic, high = creative] )
  * Greedy (highest probability)
  * Random (weighted) - random-weighted strategy across the probabilities of all tokens
  * Top-k (constrain sampling to most probable tokens)
  * Top-p (constrain to whose cumulative probabilities do not exceed threshold)


# 5. Generative Adversarial Netoworks (GANs)
## 5.1 Training Dynamics
   - Generator:
        * Creates synthetic samples (e.g., fake images)
   - Discriminator:
        * Learns to distinguish real/fake


# 6. Retrieval-Augemented Generation (RAG)
## 6.3 Why RAG?
  * Fine-tuning a pre-trained LLM is still resource-intensive and expensive.
  * Private content might change frequently, this means fine-tuning needs to be done every time content changes
  * (insert screenshot)

## 6.2 Architecture
  * Retriever:
      - Searches knowledge base (e.g., vector DB of documents)
  * Generator:
      - LLM synthesizes retrieved info + query
## 6.3 Business Use Cases
  * Customer Support:
      - Pulls relevant policy docs before answering
  * Medical Q&A:
      - Grounds responses in latest research ?????

    
