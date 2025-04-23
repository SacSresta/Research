import subprocess

def run_scripts():
    print("Running scripts...")
    scripts = ['data_fetcher.py', 'categorical_main_date.py']
    
    for script in scripts:
        print(f"Starting {script}")
        subprocess.run(['python', script])
        print(f"Completed {script}")

if __name__ == "__main__":
    run_scripts()
