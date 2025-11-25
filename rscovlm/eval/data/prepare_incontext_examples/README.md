### In-Context Learning: Selecting samples from traning set
- Setting 1 Random (online/offline): different seeds for three runs and average the results (if they differ a lot, take ten random seeds for variance)
- Setting 2 OpenFlamingo's RICES method (online): for each test sample, calculate the CLIP similarity between the image and the entire training set, and take the most similar images and answers.
- Setting 3 PPL (offline): take the samples with the highest perplexity in the entire training set for the current model