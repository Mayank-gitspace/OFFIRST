"""Train a tiny Good/Reject (or Good/Usable/Reject) classifier.
Data layout: DATA/train/<class>/*.png and DATA/val/<class>/*.png  (torchvision ImageFolder).
For EyeQ: sort images into good/usable/reject folders using the repo's label CSVs.
Usage: python train.py DATA --epochs 8 --out qg.pt
"""
import argparse, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224); ap.add_argument("--out", default="qg.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # NOTE: no blur/brightness augmentation - those are exactly what we're detecting.
    tr = transforms.Compose([transforms.Resize((a.size, a.size)), transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(15), transforms.ToTensor(), norm])
    te = transforms.Compose([transforms.Resize((a.size, a.size)), transforms.ToTensor(), norm])
    dtr = datasets.ImageFolder(f"{a.data}/train", tr); dva = datasets.ImageFolder(f"{a.data}/val", te)
    print("classes:", dtr.classes)
    counts = torch.bincount(torch.tensor(dtr.targets), minlength=len(dtr.classes)).float()
    weights = (counts.sum() / (len(counts) * counts)).to(dev)  # class-weighted loss
    ltr = DataLoader(dtr, a.bs, shuffle=True, num_workers=4); lva = DataLoader(dva, a.bs, num_workers=4)

    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(dtr.classes)); m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss(weight=weights)
    reject = dtr.class_to_idx.get("reject")
    best = 0.0
    for ep in range(a.epochs):
        m.train(); t0 = time.time()
        for x, y in ltr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
        sched.step(); m.eval(); correct = n = tp = pos = 0
        with torch.no_grad():
            for x, y in lva:
                p = m(x.to(dev)).argmax(1).cpu(); correct += (p == y).sum().item(); n += len(y)
                if reject is not None:
                    pos += (y == reject).sum().item(); tp += ((p == reject) & (y == reject)).sum().item()
        acc = correct / n; rrec = tp / pos if pos else float("nan")
        print(f"epoch {ep+1}: val acc {acc:.4f}  reject recall {rrec:.4f}  ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc; torch.save({"state": m.state_dict(), "classes": dtr.classes, "size": a.size}, a.out)
    print("best val acc", best, "saved", a.out)

if __name__ == "__main__":
    main()
