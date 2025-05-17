import torch, glob
for f in glob.glob("gpt2_realignr_resume_step*.pth"):
    l2 = torch.load(f, map_location="cpu")["model_state_dict"]["tok_emb.weight"][0,:5].norm().item()
    print(f"{f:50s}  L2 norm = {l2:.2f}")
