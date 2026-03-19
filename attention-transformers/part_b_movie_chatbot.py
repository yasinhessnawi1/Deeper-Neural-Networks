"""
Assignment 2 - Part B: Transformer-Based Movie Chatbot
=======================================================
Uses a pretrained T5 model fine-tuned (minimally) on a small
IMDb-style movie knowledge corpus to answer movie questions.

Pipeline:
  1. Build a small movie QA dataset from IMDb-style plot summaries
  2. Load a pretrained T5-small model
  3. Perform minimal fine-tuning (few epochs, NOT to convergence)
  4. Implement a simple chatbot interface
  5. Test with 5+ questions & analyse failures / hallucinations
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

# ── 1. Small Movie Knowledge Corpus ─────────────────────────────────────────
# IMDb-style plot summaries / facts (hand-curated small subset)

MOVIE_CORPUS = [
    # Star Wars
    {
        "plot": "Star Wars: A New Hope is a 1977 epic space opera film. Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire's world-destroying battle station, the Death Star. The main villain is Darth Vader, a Sith Lord who serves the Galactic Empire. The film was directed by George Lucas.",
        "qa_pairs": [
            ("Who is the villain in Star Wars?", "Darth Vader"),
            ("Who directed Star Wars: A New Hope?", "George Lucas"),
            ("What is the Death Star?", "The Empire's world-destroying battle station"),
            ("Who is the main hero of Star Wars?", "Luke Skywalker"),
        ]
    },
    # The Godfather
    {
        "plot": "The Godfather is a 1972 crime film directed by Francis Ford Coppola. It tells the story of the Corleone family, a powerful Italian-American Mafia clan. Vito Corleone is the aging patriarch and Don of the family. His youngest son Michael Corleone reluctantly joins the family business and eventually becomes the new Don.",
        "qa_pairs": [
            ("Who directed The Godfather?", "Francis Ford Coppola"),
            ("Who is the patriarch in The Godfather?", "Vito Corleone"),
            ("Who becomes the new Don in The Godfather?", "Michael Corleone"),
            ("What year was The Godfather released?", "1972"),
        ]
    },
    # The Dark Knight
    {
        "plot": "The Dark Knight is a 2008 superhero film directed by Christopher Nolan. Batman, whose real identity is Bruce Wayne, faces the Joker, a criminal mastermind who wants to plunge Gotham City into chaos. Harvey Dent, the district attorney, is corrupted and becomes the villain Two-Face. The film stars Christian Bale as Batman and Heath Ledger as the Joker.",
        "qa_pairs": [
            ("Who is the villain in The Dark Knight?", "The Joker"),
            ("Who directed The Dark Knight?", "Christopher Nolan"),
            ("Who plays Batman in The Dark Knight?", "Christian Bale"),
            ("Who plays the Joker in The Dark Knight?", "Heath Ledger"),
            ("What is Batman's real identity?", "Bruce Wayne"),
        ]
    },
    # Inception
    {
        "plot": "Inception is a 2010 science fiction film directed by Christopher Nolan. Dom Cobb, played by Leonardo DiCaprio, is a thief who steals secrets from people's dreams. He is offered a chance to have his criminal record erased if he can perform inception: planting an idea in someone's mind through their dreams.",
        "qa_pairs": [
            ("Who directed Inception?", "Christopher Nolan"),
            ("Who stars in Inception?", "Leonardo DiCaprio"),
            ("What is inception in the movie?", "Planting an idea in someone's mind through their dreams"),
            ("What does Dom Cobb do?", "He is a thief who steals secrets from people's dreams"),
        ]
    },
    # Titanic
    {
        "plot": "Titanic is a 1997 epic romance and disaster film directed by James Cameron. It tells the love story of Jack Dawson, played by Leonardo DiCaprio, and Rose DeWitt Bukater, played by Kate Winslet, aboard the ill-fated RMS Titanic. The ship hits an iceberg and sinks. Jack dies in the freezing water.",
        "qa_pairs": [
            ("Who directed Titanic?", "James Cameron"),
            ("Who plays Jack in Titanic?", "Leonardo DiCaprio"),
            ("Who plays Rose in Titanic?", "Kate Winslet"),
            ("What happens to the Titanic?", "It hits an iceberg and sinks"),
        ]
    },
    # Pulp Fiction
    {
        "plot": "Pulp Fiction is a 1994 crime film directed by Quentin Tarantino. It features interconnected stories of criminals in Los Angeles. The film stars John Travolta as Vincent Vega, Samuel L. Jackson as Jules Winnfield, and Uma Thurman as Mia Wallace. It is known for its nonlinear narrative structure.",
        "qa_pairs": [
            ("Who directed Pulp Fiction?", "Quentin Tarantino"),
            ("Who plays Vincent Vega in Pulp Fiction?", "John Travolta"),
            ("What is Pulp Fiction known for?", "Its nonlinear narrative structure"),
        ]
    },
]


# ── 2. Build QA Dataset ─────────────────────────────────────────────────────

class MovieQADataset(Dataset):
    """Simple QA dataset: input = 'question: ... context: ...' → target = answer."""

    def __init__(self, corpus, tokenizer, max_input_len=256, max_target_len=64):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        for movie in corpus:
            plot = movie["plot"]
            for question, answer in movie["qa_pairs"]:
                input_text = f"question: {question}  context: {plot}"
                self.examples.append((input_text, answer))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, target_text = self.examples[idx]

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Replace pad token id with -100 so loss ignores padding
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


# ── 3. Fine-tuning loop (minimal) ───────────────────────────────────────────

def fine_tune(model, dataloader, optimizer, device, epochs=3):
    """Minimal fine-tuning — intentionally few epochs, NOT to convergence."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch + 1}/{epochs}  —  avg loss: {avg_loss:.4f}")
    return model


# ── 4. Chatbot interface ────────────────────────────────────────────────────

class MovieChatbot:
    """Simple movie QA chatbot backed by a fine-tuned T5 model."""

    def __init__(self, model, tokenizer, corpus, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Pre-build a context string from all plots
        self.context = " ".join(m["plot"] for m in corpus)

    def generate(self, question, max_length=64):
        """Answer a question using the fine-tuned model."""
        input_text = f"question: {question}  context: {self.context}"
        input_enc = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                **input_enc,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer


# ── 5. Main execution ───────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Part B: Transformer-Based Movie Chatbot (T5-small)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load pretrained T5-small
    print("\n[1/4] Loading pretrained T5-small model & tokenizer ...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Model parameters: {total_params:,}")

    # Build dataset
    print("\n[2/4] Building movie QA dataset ...")
    dataset = MovieQADataset(MOVIE_CORPUS, tokenizer)
    print(f"       Total QA pairs: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fine-tune (minimal — 3 epochs only)
    print("\n[3/4] Minimal fine-tuning (3 epochs, NOT to convergence) ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model = fine_tune(model, dataloader, optimizer, device, epochs=3)

    # Build chatbot
    print("\n[4/4] Chatbot ready!\n")
    chatbot = MovieChatbot(model, tokenizer, MOVIE_CORPUS, device)

    # ── 6. Test with 5+ questions ────────────────────────────────────────────
    test_questions = [
        "Who is the villain in Star Wars?",
        "What movie features Darth Vader?",
        "Who directed The Godfather?",
        "Who plays the Joker in The Dark Knight?",
        "What happens to the Titanic?",
        "Who stars in Inception?",
        "Who directed Pulp Fiction?",
        # Questions likely to expose failures / hallucination
        "Who is the villain in Titanic?",          # no clear villain in our data
        "What year was Inception released?",       # answer is in the plot
        "Who plays Mia Wallace in Pulp Fiction?",  # Uma Thurman
    ]

    print("=" * 65)
    print("  CHATBOT TEST RESULTS")
    print("=" * 65)
    for q in test_questions:
        answer = chatbot.generate(q)
        print(f"\n  Q: {q}")
        print(f"  A: {answer}")

    # ── 7. Analysis ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  ANALYSIS")
    print("=" * 65)
    print("""
    1. WHERE DOES THE MODEL FAIL?
       - The model was only minimally fine-tuned (3 epochs), so it may
         give incomplete or slightly off answers, especially for questions
         whose phrasing differs from the training QA pairs.
       - Questions about information NOT present in the corpus (e.g.,
         "Who is the villain in Titanic?") will likely produce incorrect
         or vague answers because there is no villain explicitly named
         in our Titanic plot summary.

    2. DOES IT HALLUCINATE?
       - Yes. With minimal fine-tuning, the model may "hallucinate" —
         confidently produce answers that are plausible but not grounded
         in the provided context. For instance, it might name a character
         from the wrong movie or invent details.
       - T5 retains knowledge from its pretraining corpus, so some
         answers may come from pretrained knowledge rather than our
         small fine-tuning dataset.

    3. HOW DOES ATTENTION HELP IN LONG CONTEXTS?
       - Self-attention allows every token to attend to every other token
         in the input, regardless of distance. This means the model can
         directly connect a question word ("villain") to a relevant word
         far away in the context ("Darth Vader") without passing through
         intermediate tokens (as RNNs must do).
       - The multi-head attention mechanism lets the model capture
         different types of relationships (syntactic, semantic, positional)
         simultaneously across the input.

    4. WHAT LIMITATIONS COME FROM THE DATASET vs THE MODEL?
       Dataset limitations:
         - Very small corpus (6 movies, ~24 QA pairs)
         - Limited phrasing variety — the model has seen each question
           pattern only once
         - Missing information for some queries (no villain in Titanic)

       Model limitations:
         - T5-small (60M params) has limited capacity compared to larger
           variants (T5-base, T5-large)
         - Minimal fine-tuning means the model hasn't fully adapted to
           the QA format
         - Input truncation at 512 tokens may cut off parts of the
           combined context when all plots are concatenated
    """)

    # ── 8. Interactive mode (optional) ───────────────────────────────────────
    print("\n" + "-" * 65)
    print("  Interactive mode — type a question or 'quit' to exit")
    print("-" * 65)
    while True:
        try:
            question = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        answer = chatbot.generate(question)
        print(f"  Bot: {answer}")


if __name__ == "__main__":
    main()
