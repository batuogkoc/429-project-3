import os
import re
import csv

def parse_filename(filename):
    match = re.search(r'n(\d+)-t(\d+)-y(\d+)-x(\d+)', filename)
    if match:
        return match.groups()
    return None

def extract_data_from_file(filepath):
    with open(filepath, 'r') as file:
        contents = file.read()
        gflops_rate_match = re.search(r'Sustained Gflops Rate\s*:\s*(\d+\.\d+)', contents)
        bandwidth_match = re.search(r'Sustained Bandwidth \(GB/sec\)\s*:\s*(\d+\.\d+)', contents)
        
        gflops_rate = gflops_rate_match.group(1) if gflops_rate_match else 'N/A'
        bandwidth = bandwidth_match.group(1) if bandwidth_match else 'N/A'
        
        return gflops_rate, bandwidth

def magic(study_index):
    csv_data = []
    for filename in os.listdir('.'):
        if filename.startswith('cardiacsim') and filename.endswith('.txt'):
            n, t, y, x = parse_filename(filename)
            gflops_rate, bandwidth = extract_data_from_file(filename)
            csv_data.append([n, t, x, y, gflops_rate, bandwidth])
    
    with open(f'output{study_index}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if study_index == 4:
            writer.writerow(['N', 'T', 'MPI', 'OMP', 'Sustained Gflops Rate', 'Sustained Bandwidth (GB/sec)'])
        else:
            writer.writerow(['N', 'T', 'X', 'Y', 'Sustained Gflops Rate', 'Sustained Bandwidth (GB/sec)'])
            writer.writerows(csv_data)
    

def main():
    os.chdir('study1')
    magic(1)
    os.chdir('..')
    os.chdir('study2')
    magic(2)
    os.chdir('..')
    os.chdir('study3')
    magic(3)
    os.chdir('..')
    # os.chdir('study4')
    # magic(4)

if __name__ == '__main__':
    main()    # BEWARE CURRENT DIRECTORY

