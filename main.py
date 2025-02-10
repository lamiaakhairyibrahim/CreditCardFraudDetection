from src.data_preprocess.data_x_y import data
import argparse

def data_org(data_org_path):
    data_input = data(path_of_data = data_org_path )
    
def main():
    parser = argparse.ArgumentParser(description="enter the dataset path.")
    parser.add_argument('path', help="The path to incloude")
    args = parser.parse_args()

    try:
        data_org(args.path)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()

