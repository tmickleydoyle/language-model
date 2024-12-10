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
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                X, Y = dataset.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
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

    # load the model
    loaded_model = GPTLanguageModel(config, dataset.vocab_size)
    loaded_model.load_state_dict(torch.load('model.pth'))
    loaded_model = loaded_model.to(config.device)

    # Example question
    question = "What is Monstera?"
    new_context = dataset.encode(question)
    print(new_context)

    # Generate sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    # print(dataset.decode(loaded_model.generate(context, max_new_tokens=500)[0].tolist()))
    print(dataset.decode(loaded_model.generate(new_context, max_new_tokens=500)[0].tolist()))

if __name__ == '__main__':
    train() 