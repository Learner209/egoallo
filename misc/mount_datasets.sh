
#!/bin/bash

# Base directory containing the datasets
base_dir="/mnt/public/datasets"

# Remote server details
remote_user="minghao"
remote_host="100.99.97.104"
remote_base_dir="/mnt/public/datasets"

# SSHFS options
sshfs_options="-o allow_other -o default_permissions -o cache=yes -o kernel_cache -o compression=no -o ServerAliveInterval=15 -o Ciphers=chacha20-poly1305@openssh.com"

# Iterate over each directory in the base directory
for dir in "$base_dir"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Get the directory name
        dir_name=$(basename "$dir")

        # Check if the directory is a mount point
        if mountpoint -q "$dir"; then
            # Check if the directory size is greater than 4 KB
            dir_size=$(du -sk "$dir" | cut -f1)
            if [ "$dir_size" -le 4 ]; then
                echo "Directory '$dir' is mounted but is empty or less than 4 KB."
                echo "Attempting to remount '$dir' using sshfs..."
                fusermount -u "$dir" 2>/dev/null
                sudo sshfs "$remote_user@$remote_host:$remote_base_dir/$dir_name" "$dir" $sshfs_options
            else
                echo "Directory '$dir' is properly mounted and contains data."
            fi
        else
            echo "Directory '$dir' is not mounted. Attempting to mount using sshfs..."
            sudo sshfs "$remote_user@$remote_host:$remote_base_dir/$dir_name" "$dir" $sshfs_options
        fi
    else
        echo "'$dir' is not a directory. Skipping..."
    fi
done
