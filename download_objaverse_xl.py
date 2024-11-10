import objaverse.xl as oxl

# Get all annotations from the dataset
annotations = oxl.get_annotations(download_dir="./objaverse")

# Sample a small number of annotations, let's say 5
sampled_annotations = annotations.sample(n=5)

# Display the sampled annotations
print(sampled_annotations)
