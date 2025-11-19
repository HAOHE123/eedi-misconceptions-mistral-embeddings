# eedi-misconceptions-mistral-embeddings

My solution/component for the Kaggle Eedi Mining Misconceptions in Mathematics competition using Mistral embeddings.

  - **Project Title**: "SFR-Embedding-Mistral for Eedi Misconceptions Competition"

  - **Description**: "Fine-tuning/using Mistral-based embeddings for ranking and recall in the Kaggle Eedi competition to map distractors to misconceptions. Involves similarity search with SimCSE, DeepSpeed, and GLORA adaptations.").

  - **Competition Link**: [Kaggle Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)

  - **Approach**: "Uses FlagEmbedding with Mistral for embedding generation, recall scripts for evaluation, and ranking on test data. Scripts like `recall.py` handle data prep, `simcse_deepspeed_mistral_glora.py` for training."

  - **Requirements**: Python 3.10+, torch, etc..

  - **How to Run**: Instructions like:

    - Install deps: `pip install -r requirements.txt`

    - Run training: `./run.sh` or `python simcse_deepspeed_mistral_glora.py`

    - Run inference: `python get_test_rank_result.py`

    - Note any args (e.g., from `run_mistral_cos_argush.sh`).
      
   
Ph.D. Candidate He Hao
https://www.linkedin.com/in/hehao123/



