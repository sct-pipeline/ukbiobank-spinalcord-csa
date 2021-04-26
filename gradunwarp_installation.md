### How to install gradunwarp

We recommend that you **do not** follow the default installation instructions for `gradunwarp`, due to the reasons outlined [here](https://github.com/sct-pipeline/ukbiobank-spinalcord-csa/issues/31).

Instead, we provide our own set of steps to help make the installation process smoother.

1. Install the dependencies of `gradunwarp`

```bash
pip install numpy scipy nibabel
```

2. Download and extract the [latest release](https://github.com/Washington-University/gradunwarp/releases/latest) of `gradunwarp`
   

3. Navigate to the extracted folder, then install `gradunwarp`

```bash
cd gradunwarp-1.2.0/
python setup.py install --prefix="~/.local"
```

4. Update your `$PATH` to include the installation location

```bash
# For bash users
echo "export PATH=$PATH:~/.local/bin" >> ~/.bashrc

# For zsh users
echo "export PATH=$PATH:~/.local/bin" >> ~/.zshrc
```
   
5. You're done! The command `gradient_unwarp.py` should now be available in your terminal after restarting the terminal.