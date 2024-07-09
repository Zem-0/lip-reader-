import os

def extract_real_words(alignments_dir, extracted_words_dir):
    # Ensure extracted words directory exists, create if not
    os.makedirs(extracted_words_dir, exist_ok=True)

    # Iterate through each file in the alignments directory
    for filename in os.listdir(alignments_dir):
        if filename.endswith('.align'):  # Assuming alignment files end with .align
            alignment_path = os.path.join(alignments_dir, filename)
            words = []

            # Read words from the alignment file
            with open(alignment_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) > 2:
                        words.append(parts[2])  # Assuming the word is in the third column

            # Write words to a separate text file for each alignment file
            output_file = os.path.join(extracted_words_dir, f"{filename.split('.')[0]}_words.txt")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.write('\n'.join(words))
# Example usage:
# Replace with your actual paths
alignments_dir = 'data\\alignments\\s1'
extracted_words_dir = 'align'

extract_real_words(alignments_dir, extracted_words_dir)