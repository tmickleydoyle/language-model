import torch
from config import Config
from data import TextDataset
from model import GPTLanguageModel

def train():
    # Initialize configuration
    config = Config()
    
    # Load and prepare data
    dataset = TextDataset(config)
    dataset.load_data('input.txt')
    
    # Initialize model with actual vocabulary size from dataset
    model = GPTLanguageModel(config, dataset.vocab_size)
    model = model.to(config.device)
    print(f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_iters, device=config.device)
            for k in range(config.eval_iters):
                X, Y = dataset.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out
    
    # Training loop
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = dataset.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    # Load the model for inference
    loaded_model = GPTLanguageModel(config, dataset.vocab_size)
    loaded_model.load_state_dict(torch.load('model.pth'))
    loaded_model = loaded_model.to(config.device)

    # Example question
    question = "What is Monstera?"
    new_context = dataset.encode(question)
    print("Encoded question:", new_context)

    # Ensure new_context is a tensor and has a batch dimension (i.e. shape (1, T))
    if not isinstance(new_context, torch.Tensor):
        new_context = torch.tensor(new_context, dtype=torch.long, device=config.device)
    if new_context.dim() == 1:
        new_context = new_context.unsqueeze(0)  # add batch dimension

    # Generate sample text using the loaded model
    generated_tokens = loaded_model.generate(new_context, max_new_tokens=500)[0].tolist()
    decoded_text = dataset.decode(generated_tokens)
    print("Generated text:")
    print(decoded_text)

if __name__ == '__main__':
    train()
