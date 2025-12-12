import datetime
import os
import yaml

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def getTime() -> str:
    return f"{bcolors.OKBLUE}[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]{bcolors.ENDC}"

def load_config(path):
    try:
        with open(path, "r") as stream:
            config = yaml.safe_load(stream)

        if config is None:
            raise ValueError("Config file is empty.")

        return config

    except FileNotFoundError as error:
        print(f"{getTime()}{bcolors.FAIL} ❌ Config file not found: '{path}' {bcolors.ENDC}{error}")
        raise

    except yaml.YAMLError as error:
        print(f"{getTime()}{bcolors.FAIL} ❌ YAML parsing error in config file '{path}': {bcolors.ENDC}{error}")
        raise

    except Exception as error:
        print(f"{getTime()}{bcolors.FAIL} ❌ Unexpected error while loading config '{path}': {bcolors.ENDC}{error}")
        raise


def welcome_message(name: str, version: str, report_title: str, config_path: str, reports_path: str):
    num_equals = 60 # Number of '=' to print
    
    message = f"WELCOME TO {name} {version}"
    
    try:
        def msgError(category: str):
            return  f"\nCONFIG ERROR: {category} must be a string. Make sure it is enclosed in double quotes in {config_path}.\n"
        
        if(isinstance(name, str) == False):
            raise TypeError(msgError("Name"))
        if(isinstance(version, str) == False):
            raise TypeError(msgError("Version"))

        if(num_equals > len(message)):
            equals_message = (num_equals - len(message)) / 2
            if(equals_message.is_integer() == False):
                equals_message = int(equals_message)
                num_equals += 1
        else:
            equals_message = 0
        
        # Output
        print("\n"+"="*num_equals)
        print("="*equals_message + " " + message + " " + "="*equals_message)
        print("="*num_equals)

        report_final_dir = f"{reports_path}{report_title}"

        def verificateion_dir(dir: str):
            if(os.path.exists(dir) == False):
                os.mkdir(dir)
                os.mkdir(f"sample_data/{report_title}")
            else:
                print(f"\n{getTime()} {bcolors.WARNING}WARNING: Report Directory '{dir}' already exists. Do you want to overwrite (Y) or choose another name (N)? \n")
                user_input = input(f"{bcolors.ENDC}Enter Y or N: ").strip().upper()
                if(user_input == "N"):
                    new_title = input("Enter new report title: ").strip()
                    r_title = new_title
                    r_final_dir = f"{reports_path}{r_title}"
                    verificateion_dir(r_final_dir)
                elif(user_input == "Y"):
                    print(f"\n{getTime()} {bcolors.WARNING}WARNING: Overwriting Report Directory '{dir}'. Previous data may be lost.{bcolors.ENDC}\n")

        verificateion_dir(report_final_dir)
        print(f"\n{getTime()} System Correctly Loaded. Report Title: {report_title}\n")

    except Exception as error:
        print(error)
