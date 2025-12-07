#!/usr/bin/env python3
"""
PPM Image Comparison Utility
Compares two PPM images pixel-by-pixel with tolerance for floating-point variations.
"""

import sys
import struct

def parse_ppm(filename):
    """Parse a PPM file and return header info and pixel data."""
    with open(filename, 'rb') as f:
        # Read magic number
        magic = f.readline().decode('ascii').strip()
        if magic not in ['P3', 'P6']:
            raise ValueError(f"Unsupported PPM format: {magic}")
        
        # Skip comments
        line = f.readline().decode('ascii')
        while line.startswith('#'):
            line = f.readline().decode('ascii')
        
        # Read dimensions
        width, height = map(int, line.strip().split())
        
        # Read max value
        maxval = int(f.readline().decode('ascii').strip())
        
        # Read pixel data
        if magic == 'P3':
            # ASCII format
            data = f.read().decode('ascii').split()
            pixels = [int(x) for x in data]
        else:
            # Binary format (P6)
            data = f.read()
            if maxval < 256:
                pixels = list(data)
            else:
                pixels = [struct.unpack('>H', data[i:i+2])[0] for i in range(0, len(data), 2)]
        
        return {
            'magic': magic,
            'width': width,
            'height': height,
            'maxval': maxval,
            'pixels': pixels
        }

def compare_images(img1_path, img2_path, tolerance_percent=1.0):
    """
    Compare two PPM images with tolerance for minor variations.
    Returns (match, difference_percent, message)
    """
    try:
        img1 = parse_ppm(img1_path)
        img2 = parse_ppm(img2_path)
    except Exception as e:
        return False, 100.0, f"Error reading images: {e}"
    
    # Check dimensions
    if img1['width'] != img2['width'] or img1['height'] != img2['height']:
        return False, 100.0, f"Dimension mismatch: {img1['width']}x{img1['height']} vs {img2['width']}x{img2['height']}"
    
    # Check pixel count
    if len(img1['pixels']) != len(img2['pixels']):
        return False, 100.0, f"Pixel count mismatch: {len(img1['pixels'])} vs {len(img2['pixels'])}"
    
    # Compare pixels
    total_pixels = len(img1['pixels'])
    maxval = max(img1['maxval'], img2['maxval'])
    tolerance = int(maxval * tolerance_percent / 100.0)
    
    different_pixels = 0
    total_difference = 0
    
    for i in range(total_pixels):
        diff = abs(img1['pixels'][i] - img2['pixels'][i])
        if diff > tolerance:
            different_pixels += 1
        total_difference += diff
    
    # Calculate statistics
    diff_percent = (different_pixels / total_pixels) * 100.0
    avg_diff = total_difference / total_pixels
    avg_diff_percent = (avg_diff / maxval) * 100.0
    
    # Images match if less than 0.1% of pixels differ beyond tolerance
    match = diff_percent < 0.1
    
    message = f"Diff pixels: {diff_percent:.2f}%, Avg diff: {avg_diff_percent:.2f}%"
    
    return match, diff_percent, message

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: compare_ppm.py <image1.ppm> <image2.ppm> [tolerance_percent]", file=sys.stderr)
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    match, diff_percent, message = compare_images(img1_path, img2_path, tolerance)
    
    print(message)
    sys.exit(0 if match else 1)
