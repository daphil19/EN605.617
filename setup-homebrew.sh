#!/bin/bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(~/.linuxbrew/bin/brew shellenv)"' >> ~/.profile
# this is primarily here so that we can tail it and eval, since neither copy-paste nor eval seem to work
eval "$(~/.linuxbrew/bin/brew shellenv)"