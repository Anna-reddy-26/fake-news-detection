import os
import shutil

def setup_project():
    """Setup the project structure and files"""
    
    # Create necessary directories
    directories = [
        'Dataset',
        'Dataset/DFD_manipulated_sequences',
        'Dataset/DFD_original_sequences',
        'Dataset/Images',
        'Dataset/Images/Train',
        'Dataset/Images/Test',
        'Dataset/news',
        'models',
        'static',
        'static/css',
        'static/js',
        'static/uploads',
        'templates',
        'logs',
        'results',
        'modules'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create empty files in Dataset directories (so they exist)
    empty_files = [
        'Dataset/news/Fake.csv',
        'Dataset/news/True.csv'
    ]
    
    for file in empty_files:
        with open(file, 'w') as f:
            f.write("title,text,subject,date\n")
        print(f"Created empty file: {file}")
    
    # Copy HTML templates if they exist
    html_files = ['index.html', 'upload.html', 'results.html', 'dashboard.html', 'base.html']
    for html_file in html_files:
        if os.path.exists(html_file):
            shutil.copy(html_file, f'templates/{html_file}')
            print(f"Copied template: {html_file}")
    
    print("\n" + "="*50)
    print("Project setup completed!")
    print("="*50)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train models: python train.py --model all")
    print("3. Run the app: python app.py")
    print("\nNote: If you don't have a MongoDB server, the app will")
    print("still work but won't save detection history.")
    print("="*50)

if __name__ == "__main__":
    setup_project()