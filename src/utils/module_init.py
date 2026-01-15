"""
Utility to initialize Compute Canada module system environment in Python.

Since Python can't directly source shell scripts, this module provides functions
to set up the module system environment by reading and executing profile scripts.
"""

import os
import subprocess
import shlex


def source_profile_and_get_env(profile_path="/etc/profile"):
    """
    Source a shell profile script and return the environment variables.
    
    This runs the profile script in a shell and captures the resulting
    environment variables.
    
    Args:
        profile_path: Path to the profile script to source
        
    Returns:
        dict: Dictionary of environment variables after sourcing the profile
    """
    # Use bash to source the profile and export all variables
    cmd = f'source {profile_path} && env'
    
    try:
        # Run in a shell and capture the environment
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"Warning: Failed to source {profile_path}: {result.stderr}")
            return {}
        
        # Parse the environment variables
        env_dict = {}
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                env_dict[key] = value
        
        return env_dict
    except Exception as e:
        print(f"Error sourcing profile: {e}")
        return {}


def init_module_system():
    """
    Initialize the Compute Canada module system environment.
    
    This attempts to set up the module system by:
    1. Sourcing /etc/profile to get base environment
    2. Setting up module-related environment variables
    3. Adding module system paths to PATH
    
    Returns:
        bool: True if module system was initialized, False otherwise
    """
    # Try to source /etc/profile
    env_vars = source_profile_and_get_env("/etc/profile")
    
    if not env_vars:
        # Fallback: try to find module system paths directly
        module_sh_paths = [
            "/etc/profile.d/modules.sh",
            "/usr/share/modules/init/bash",
            "/opt/software/modules/init/bash",
        ]
        
        for module_sh in module_sh_paths:
            if os.path.exists(module_sh):
                env_vars = source_profile_and_get_env(module_sh)
                if env_vars:
                    break
    
    # Update current environment with the sourced variables
    updated = False
    for key, value in env_vars.items():
        if key not in os.environ or os.environ[key] != value:
            os.environ[key] = value
            updated = True
    
    return updated


def run_with_module_command(cmd):
    """
    Run a module command (e.g., 'module load cuda/12.2') by sourcing
    the profile and then executing the command in a shell.
    
    Args:
        cmd: Module command to run (e.g., "module load cuda/12.2")
        
    Returns:
        dict: Environment variables after running the module command
    """
    # Construct command to source profile and run module command
    full_cmd = f'source /etc/profile && {cmd} && env'
    
    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Warning: Module command failed: {result.stderr}")
            return {}
        
        # Parse environment variables
        env_dict = {}
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                env_dict[key] = value
        
        # Update current process environment
        for key, value in env_dict.items():
            os.environ[key] = value
        
        return env_dict
    except Exception as e:
        print(f"Error running module command: {e}")
        return {}


def load_module(module_name):
    """
    Load a module and update the current Python process environment.
    
    Args:
        module_name: Name of module to load (e.g., "cuda/12.2")
        
    Returns:
        bool: True if module was loaded successfully, False otherwise
    """
    env_vars = run_with_module_command(f"module load {module_name}")
    return len(env_vars) > 0


# Alternative: Direct approach for Compute Canada systems
def init_computecanada_modules():
    """
    Initialize Compute Canada module system by directly setting common paths.
    
    This is a more direct approach that doesn't require sourcing shell scripts.
    """
    # Common Compute Canada module system paths
    module_paths = [
        "/cvmfs/soft.computecanada.ca/config/modules",
        "/opt/software/modules",
    ]
    
    # Add to MODULEPATH if it exists
    modulepath = os.environ.get("MODULEPATH", "")
    for path in module_paths:
        if os.path.exists(path) and path not in modulepath:
            if modulepath:
                modulepath = f"{path}:{modulepath}"
            else:
                modulepath = path
    
    if modulepath:
        os.environ["MODULEPATH"] = modulepath
    
    # Try to find and set up module command
    # Note: The actual module command is a shell function, not a binary
    # So we can't directly call it from Python, but we can set up the environment
    # so that when we run subprocess commands, they can use modules
    
    return modulepath != ""


if __name__ == "__main__":
    # Test the functions
    print("Testing module system initialization...")
    
    # Method 1: Try to source profile
    print("\n1. Attempting to source /etc/profile...")
    env_vars = source_profile_and_get_env("/etc/profile")
    print(f"   Found {len(env_vars)} environment variables")
    if "MODULEPATH" in env_vars:
        print(f"   MODULEPATH: {env_vars['MODULEPATH'][:100]}...")
    
    # Method 2: Initialize module system
    print("\n2. Initializing module system...")
    init_module_system()
    print(f"   MODULEPATH: {os.environ.get('MODULEPATH', 'Not set')[:100]}...")
    
    # Method 3: Try to load a module
    print("\n3. Testing module load...")
    if load_module("cuda/12.2"):
        print("   ✓ Successfully loaded cuda/12.2")
        print(f"   CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    else:
        print("   ✗ Failed to load module (this is expected if module command isn't available)")
