import pandas as pd
import random
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def generate_synthetic_data(input_csv="train_data.csv", output_csv="questions_synthetic.csv", num_samples=20):
    df = pd.read_csv(input_csv)
    
    # Select random documents
    sampled_df = df.sample(n=min(num_samples, len(df)))
    
    client = OpenAI(
        base_url="https://api.perplexity.ai",
        api_key=os.getenv("LLM_API_KEY")
    )
    
    new_data = []
    
    print("Generating synthetic questions...")
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        context = f"{row['annotation']} {row['text'][:500]}"
        
        prompt = (
            f"Context: {context}\n\n"
            "Generate a simple, factual question based ONLY on this text. "
            "Output ONLY the question."
        )
        
        try:
            resp = client.chat.completions.create(
                model="sonar",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            question = resp.choices[0].message.content.strip()
            
            new_data.append({
                "question": question,
                "ground_truth_doc_id": idx,  # We know the answer comes from this doc
                "ground_truth_text": context
            })
        except Exception as e:
            print(f"Error: {e}")
            
    # Save
    out_df = pd.DataFrame(new_data)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} synthetic questions to {output_csv}")

if __name__ == "__main__":
    generate_synthetic_data()
