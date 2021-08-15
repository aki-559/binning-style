import argparse
import torch
import torch.optim as optim
import numpy as np
from model import Discriminator, train, test
from loader import DataLoader, read_all, to_tensor
from Bio import SeqIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer", help="layer for calculating style matrices", type=int, default=4)
    parser.add_argument("-e", "--epoch", help="epoch for training a model", type=int, default=100)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=0.001)
    parser.add_argument("-d", "--dir", help="directory that contains fasta files for training", default="./data")
    parser.add_argument("-c", "--contig", help="a fasta file that contains contigs", default="./test.fna")
    parser.add_argument("-o", "--output", help="an output file for extracted style matrices", default="./genome_style.pt")
    parser.add_argument("--verbose", help="logging level", type=int, default=2)
    parser.add_argument("--batch-size", help="batch size", type=int, default=64)
    parser.add_argument("--n-steps", help="number of steps per epoch", type=int, default=500)
    parser.add_argument("--length", help="length of sampled sequence", type=int, default=1024)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Read data
    if args.verbose > 1: print("Reading fasta...")
    species, seqs, labels = read_all(args.dir)

    loader = DataLoader(batch_size=args.batch_size, use_all=True, n_batch=args.n_steps)
    loader(species, seqs, labels)

    # train model
    if args.verbose > 1: print("\nTraining model...")
    
    model = Discriminator(args.length, len(species)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.rate)

    for epoch in range(args.epoch):
        train(model, device, loader, optimizer, epoch+1)
        print("")

    # calculate style matrices
    if args.verbose > 1: print("Extracting style matrices...")
    
    model.eval()
    style_matrices = []

    for record in SeqIO.parse(args.contig, "fasta"):
        tensor = to_tensor(str(record.seq))
        style_matrices += model.get_style(tensor.float().to(device), args.layer)

    style_matrices = torch.cat(style_matrices, dim=0)

    torch.save(style_matrices, args.output)

    if args.verbose > 1: print("Genome style matrix is successfully written to {}.".format(args.output))