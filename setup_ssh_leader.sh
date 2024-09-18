#!/bin/bash

#sudo apt-get update
#sudo apt-get install sshpass

# Define the member nodes and their passwords
declare -A MEMBER_NODES
MEMBER_NODES=(
    ["member1"]="password1"
    ["member2"]="password2"
    ["member3"]="password3"
)  # Add your member nodes and their passwords here

# Step 1: Disable strictHostKeyChecking and enable ForwardAgent
mkdir -p ~/.ssh
echo -e "Host *\n    ForwardAgent yes\nHost *\n    StrictHostKeyChecking no" > ~/.ssh/config
chmod 600 ~/.ssh/config

# Step 2: Generate an RSA key pair
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa

# Step 3: Change permissions of the private key
chmod 600 ~/.ssh/id_rsa

# Step 4: Add the public key to the leader node's authorized_keys
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Step 5: Copy the public key to each member node
for MEMBER in "${!MEMBER_NODES[@]}"; do
    echo "Copying SSH key to $MEMBER"
    sshpass -p "${MEMBER_NODES[$MEMBER]}" ssh -o StrictHostKeyChecking=no $MEMBER 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub
done

echo "Passwordless SSH setup complete."

#chmod +x setup_ssh_leader.sh
#./setup_ssh_leader.sh