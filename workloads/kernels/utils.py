import os
import numpy as np
import torch

def save_matrices_to_txt(*matrices, 
                        output_dir: str = ".", 
                        precision: int = 9,
                        dtype: str = "float",
                        prefix: str = "matrix",
                        delimiter: str = " ",
                        create_manifest: bool = False,
                        start_idx: int = 1):
    """
    Save multiple torch tensors or numpy arrays to txt files with intelligent flattening.
    
    Args:
        *matrices: Variable number of torch tensors or numpy arrays
        output_dir: Directory to save the txt files (default: current directory)
        precision: Number of decimal places for floating point numbers (default: 6)
        dtype: Data type for formatting ("float", "int", "scientific", "auto") (default: "float")
        prefix: Prefix for output filenames (default: "matrix")
        delimiter: Delimiter between values in the same row (default: " ")
        create_manifest: Whether to create a manifest file with metadata (default: False)
        start_idx: Starting index for filename numbering (default: 1)
    
    Returns:
        List of saved file paths
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    manifest_data = []
    
    for i, matrix in enumerate(matrices):
        idx = start_idx + i  # Use custom starting index
        
        # Convert to numpy if it's a torch tensor
        if isinstance(matrix, torch.Tensor):
            data = matrix.detach().cpu().numpy()
            original_type = "torch.Tensor"
            original_device = str(matrix.device)
            original_dtype = str(matrix.dtype)
        elif isinstance(matrix, np.ndarray):
            data = matrix
            original_type = "numpy.ndarray"
            original_device = "cpu"
            original_dtype = str(matrix.dtype)
        else:
            raise TypeError(f"Parameter {idx} must be a torch.Tensor or numpy.ndarray, "
                          f"got {type(matrix)}")
        
        # Get original shape for filename and metadata
        original_shape = data.shape
        
        # Handle shape string generation properly for 1D arrays
        if len(original_shape) == 1:
            # For 1D arrays like (768,), make it (1, 768) in the filename
            shape_str = f"1x{original_shape[0]}"
            display_shape = f"(1, {original_shape[0]})"
        else:
            shape_str = "x".join(map(str, original_shape))
            display_shape = str(original_shape)

        # Flatten all dimensions except the last one
        if len(data.shape) > 1:
            # Reshape to (product_of_all_except_last, last_dimension)
            new_shape = (-1, data.shape[-1])
            flattened_data = data.reshape(new_shape)
        else:
            # If 1D, reshape to (1, length) for consistency - 1 row, M columns
            flattened_data = data.reshape(1, -1)
        
        rows, cols = flattened_data.shape
        
        # Generate filename with custom index
        filename = f"{prefix}_{shape_str}_{idx}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Auto-detect data type if specified
        if dtype.lower() == "auto":
            if np.issubdtype(data.dtype, np.integer):
                format_str = "%d"
                detected_dtype = "int"
            elif np.issubdtype(data.dtype, np.floating):
                # Check if values are very large or very small
                abs_data = np.abs(data[np.isfinite(data)])
                if len(abs_data) > 0:
                    max_val = np.max(abs_data)
                    min_val = np.min(abs_data[abs_data > 0]) if np.any(abs_data > 0) else 0
                    if max_val > 10**6 or (min_val > 0 and min_val < 10**(-4)):
                        format_str = f"%.{precision}e"
                        detected_dtype = "scientific"
                    else:
                        format_str = f"%.{precision}f"
                        detected_dtype = "float"
                else:
                    format_str = f"%.{precision}f"
                    detected_dtype = "float"
            else:
                format_str = f"%.{precision}f"
                detected_dtype = "float"
        else:
            # Use specified dtype
            if dtype.lower() == "int":
                format_str = "%d"
                detected_dtype = "int"
            elif dtype.lower() == "scientific":
                format_str = f"%.{precision}e"
                detected_dtype = "scientific"
            else:  # default to float
                format_str = f"%.{precision}f"
                detected_dtype = "float"
        
        # Save to file
        with open(filepath, 'w') as f:
            # Write dimensions as first line
            f.write(f"{rows}{delimiter}{cols}\n")
            
            # Write data
            for row in flattened_data:
                if len(row) == 1:
                    # Single column
                    f.write(format_str % row[0] + "\n")
                else:
                    # Multiple columns
                    row_str = delimiter.join([format_str % val for val in row])
                    f.write(row_str + "\n")
        
        saved_files.append(filepath)
        
        # Collect metadata for manifest
        manifest_entry = {
            'index': idx,
            'filename': filename,
            'original_shape': original_shape,
            'flattened_shape': (rows, cols),
            'original_type': original_type,
            'original_dtype': original_dtype,
            'original_device': original_device,
            'saved_dtype': detected_dtype,
            'precision': precision,
            'delimiter': delimiter
        }
        manifest_data.append(manifest_entry)
        
        print(f"Saved matrix {idx}: {original_shape} -> {display_shape} "
              f"(flattened to {rows}x{cols}) as {detected_dtype} to: {filepath}")
    
    # Create manifest file
    if create_manifest and saved_files:
        manifest_path = os.path.join(output_dir, f"{prefix}_manifest.txt")
        with open(manifest_path, 'w') as f:
            f.write("# Matrix Save Manifest\n")
            f.write(f"# Total matrices: {len(saved_files)}\n")
            f.write("# Format: index|filename|original_shape|flattened_shape|type|dtype|device|saved_dtype|precision|delimiter\n")
            for entry in manifest_data:
                f.write(f"{entry['index']}|{entry['filename']}|{entry['original_shape']}|"
                       f"{entry['flattened_shape']}|{entry['original_type']}|{entry['original_dtype']}|"
                       f"{entry['original_device']}|{entry['saved_dtype']}|{entry['precision']}|"
                       f"'{entry['delimiter']}'\n")
        print(f"Created manifest file: {manifest_path}")
    
    return saved_files