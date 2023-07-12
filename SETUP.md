# Downloading this Directory
Please find a good place to store the files for this project. Creating a new folder is best so that 
no unexpected issues crop up. For example, this project is located in a folder named "PAS-BO-Interns" on my desktop. 
Once this has happened, please download all of the files in this repository and put them in that folder. 
Simply right click and hit "Download."

# Getting Started With Python
If you do not already have Python installed on your computer, you can install it in a number of ways.
For multiple reasons, such as convenience and environment management, it is usually recommended to do this through [Anaconda](https://www.anaconda.com/download). Anaconda comes with many Python packages that are useful for data science and coding in general.  
However, if you don't want to do this, you can install just Python [here](https://www.python.org/downloads/).  
These will generally require admin access to install, so you may need IT's help.
You can check if Python is installed correctly by setting up a terminal (also known as a command line).
You can do this by searching your computer for "cmd" and selecting "Command Prompt."  
When you are at this point, you should see a place where you can type commands. Type
```python -V```. If no error comes up, you are all set.  
 If you installed via Anaconda, you should also check ```conda -V```.

# Installing Packages
Packages are essentially little bundles of code that someone has already written for you in a convenient way. As a free language, Python packages are usually free and open source as well. There are a few methods of installing packages, but the most common way of doing it is to go to the command line and type ```pip install [package name]```.  
If you have Anaconda installed, you can also use ```conda install [package name]```.  
However, conveniently, you can also install a bunch of packages at once if someone has listed their names and versions in a file for you. By convention, this file is called requirements.txt, which you should be able to see in this repository. 
Open a terimnal and navigate to the directory requirements.txt is in. Navigating the command line requires the use of one command: ```cd```. If you are trying to go into a directory that is inside 
your current directory, simply type ```cd [name of subdirectory]``` and hit ENTER. If you are trying to go back to the directory that contains your current directory, type ```cd ..```.  
You can then type ```pip install -r requirements.txt```. Now you have all the packages needed!

# Running the Interface
Start a terminal and navigate to the directory all the files are located in for this project. 
Type ```python -m streamlit run testgui.py``` and hit ENTER. You should now be hosting the file!