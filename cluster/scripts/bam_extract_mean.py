import pysam
import sys
import os
import numpy as np
import csv
from concurrent.futures import ProcessPoolExecutor

class VectorizedStats:
    """Computes stats using NumPy for speed while keeping memory footprint low."""
    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def push_array(self, values):
        if not values: return
        arr = np.array(values, dtype=np.float64)
        self.n += arr.size
        self.sum_x += np.sum(arr)
        self.sum_x2 += np.sum(arr**2)

    def stats(self):
        if self.n < 1: return 0.0, 0.0
        mean = self.sum_x / self.n
        variance = (self.sum_x2 / self.n) - (mean ** 2)
        return mean, np.sqrt(max(0, variance))

def process_one_bam(bam_path):
    """Worker function: Processes a single BAM file and returns the result dictionary."""
    try:
        strain_name = os.path.basename(bam_path).replace(".bam", "")
        ipd_stats = VectorizedStats()
        pw_stats = VectorizedStats()

        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            for read in bam:
                # Process IPD tags (fi/ri)
                for tag in ['fi', 'ri']:
                    if read.has_tag(tag):
                        ipd_stats.push_array(read.get_tag(tag))
                
                # Process PW tags (fp/rp)
                for tag in ['fp', 'rp']:
                    if read.has_tag(tag):
                        pw_stats.push_array(read.get_tag(tag))

        m_ipd, s_ipd = ipd_stats.stats()
        m_pw, s_pw = pw_stats.stats()

        return {
            'strain': strain_name,
            'mean_ipd': round(m_ipd, 4), 'std_ipd': round(s_ipd, 4),
            'mean_pw': round(m_pw, 4), 'std_pw': round(s_pw, 4)
        }
    except Exception as e:
        return {'strain': os.path.basename(bam_path), 'error': str(e)}

def main():
    if len(sys.argv) < 3:
        print("Usage: python bam_extract_mean.py <list_file> <output_csv>")
        return

    list_file = sys.argv[1]
    output_csv = sys.argv[2]
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    with open(list_file, 'r') as f:
        bam_paths = [line.strip() for line in f if line.strip()]

    fieldnames = ['strain', 'mean_ipd', 'std_ipd', 'mean_pw', 'std_pw']
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        # ProcessPoolExecutor distributes the work across CPUs
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # results come back to the main thread one by one as they finish
            for result in executor.map(process_one_bam, bam_paths):
                if 'error' in result:
                    print(f"❌ Error in {result['strain']}: {result['error']}", flush=True)
                else:
                    writer.writerow(result)
                    csvfile.flush() # Write to disk immediately
                    os.fsync(csvfile.fileno()) # Force physical write
                    print(f"✅ Finished: {result['strain']}", flush=True)

if __name__ == "__main__":
    main()
