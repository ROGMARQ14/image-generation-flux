# Explicitly export required symbols
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Should exist in nodes.py
NODE_CLASS_MAPPINGS = {
    'DualCLIPLoader': DualCLIPLoader,
    'UNETLoader': UNETLoader,
    'VAELoader': VAELoader,
    # ... other mappings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'DualCLIPLoader': 'Dual CLIP Loader',
    'UNETLoader': 'UNET Loader',
    # ... other display names
}
