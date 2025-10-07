
from processor import process_dataset

# === Kjøring ===
if __name__ == "__main__":

 def figureprint_matching(approach):
     if approach == "sift_flann":
         dataset_path = r"C:\Users\dunk1\ikt213\IKT213_THEIN\figureprint_matching\data\data_check"
         results_folder = r"C:\Users\dunk1\ikt213\IKT213_THEIN\figureprint_matching\results_sift"
         process_dataset(dataset_path, results_folder, approach)
     elif approach == "orb_bf":
          dataset_path = r"C:\Users\dunk1\ikt213\IKT213_THEIN\figureprint_matching\data\data_check"
          results_folder = r"C:\Users\dunk1\ikt213\IKT213_THEIN\figureprint_matching\results_orb"
          process_dataset(dataset_path, results_folder,approach)
     else:
         print("choose approach  'sift_flann' or 'orb_bf' ")


figureprint_matching("sift_flann")
figureprint_matching("orb_bf")
