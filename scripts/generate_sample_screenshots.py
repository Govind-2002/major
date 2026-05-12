"""
Generate synthetic sample screenshots for training the visual model.
Creates placeholder images that simulate phishing and legitimate webpage screenshots.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random
import string

def generate_random_text(length=10):
    """Generate random text string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_legitimate_screenshot(path: str, brand: str):
    """Create a synthetic legitimate website screenshot."""
    # Professional color schemes for legitimate sites
    colors = {
        'google': {'bg': '#FFFFFF', 'header': '#4285F4', 'text': '#202124'},
        'microsoft': {'bg': '#FFFFFF', 'header': '#0078D4', 'text': '#323130'},
        'amazon': {'bg': '#FFFFFF', 'header': '#232F3E', 'text': '#0F1111'},
        'facebook': {'bg': '#F0F2F5', 'header': '#1877F2', 'text': '#050505'},
        'linkedin': {'bg': '#FFFFFF', 'header': '#0A66C2', 'text': '#000000'},
        'github': {'bg': '#FFFFFF', 'header': '#24292F', 'text': '#24292F'},
    }
    
    scheme = colors.get(brand, colors['google'])
    
    # Create image
    img = Image.new('RGB', (1200, 800), scheme['bg'])
    draw = ImageDraw.Draw(img)
    
    # Draw header bar
    draw.rectangle([0, 0, 1200, 60], fill=scheme['header'])
    
    # Draw URL bar (HTTPS indicator)
    draw.rectangle([100, 70, 1100, 100], fill='#F1F3F4', outline='#DADCE0')
    draw.text((120, 77), f"🔒 https://www.{brand}.com/", fill='#202124')
    
    # Draw logo placeholder
    draw.rectangle([50, 150, 200, 200], fill=scheme['header'])
    draw.text((60, 165), brand.upper()[:4], fill='#FFFFFF')
    
    # Draw content area
    draw.rectangle([50, 220, 1150, 750], fill='#FFFFFF', outline='#DADCE0')
    
    # Draw navigation
    for i, item in enumerate(['Home', 'Products', 'Services', 'Contact', 'About']):
        draw.text((250 + i*150, 165), item, fill=scheme['text'])
    
    # Draw content blocks
    for i in range(3):
        y = 250 + i*150
        draw.rectangle([70, y, 1130, y+120], fill='#F8F9FA', outline='#E8EAED')
        draw.text((90, y+20), f"Content Section {i+1}", fill=scheme['text'])
        draw.text((90, y+50), "Professional content with proper formatting and structure.", fill='#5F6368')
    
    # Draw footer
    draw.rectangle([0, 760, 1200, 800], fill='#F8F9FA')
    draw.text((500, 775), f"© 2024 {brand.title()}. All rights reserved.", fill='#5F6368')
    
    img.save(path)
    print(f"  Created: {path}")

def create_phishing_screenshot(path: str, target_brand: str, phishing_type: str):
    """Create a synthetic phishing website screenshot."""
    
    img = Image.new('RGB', (1200, 800), '#FFFFFF')
    draw = ImageDraw.Draw(img)
    
    if phishing_type == 'login_form':
        # Suspicious login form
        draw.rectangle([0, 0, 1200, 50], fill='#333333')
        
        # Suspicious URL
        draw.rectangle([100, 60, 1100, 90], fill='#FFEEEE', outline='#FF0000')
        suspicious_url = f"http://secure-{target_brand}-login.xyz/verify?token=abc123"
        draw.text((120, 67), f"⚠️ {suspicious_url}", fill='#CC0000')
        
        # Fake login box
        draw.rectangle([350, 150, 850, 550], fill='#FFFFFF', outline='#CCCCCC', width=2)
        draw.text((450, 180), f"{target_brand.title()} Login", fill='#333333')
        
        # Warning signs
        draw.rectangle([370, 230, 830, 270], fill='#F5F5F5', outline='#CCCCCC')
        draw.text((390, 243), "Email or Username", fill='#666666')
        
        draw.rectangle([370, 290, 830, 330], fill='#F5F5F5', outline='#CCCCCC')
        draw.text((390, 303), "Password", fill='#666666')
        
        # Suspicious button
        draw.rectangle([370, 360, 830, 410], fill='#FF4444')
        draw.text((550, 375), "LOGIN NOW!", fill='#FFFFFF')
        
        # Urgency message (phishing indicator)
        draw.text((380, 430), "⚠️ Your account will be suspended!", fill='#CC0000')
        draw.text((380, 460), "Verify immediately to avoid losing access!", fill='#CC0000')
        
    elif phishing_type == 'typosquatting':
        # Typosquatted domain
        draw.rectangle([0, 0, 1200, 50], fill='#1877F2')
        
        typo_name = target_brand.replace('o', '0').replace('a', '4')
        draw.rectangle([100, 60, 1100, 90], fill='#FFF3CD', outline='#FFD700')
        draw.text((120, 67), f"http://{typo_name}.com/login", fill='#856404')
        
        # Mimicked content
        draw.rectangle([300, 150, 900, 600], fill='#FFFFFF', outline='#DDDDDD')
        draw.text((420, 180), f"Welcome to {typo_name}", fill='#333333')
        
        # Poorly designed elements
        draw.rectangle([350, 250, 850, 290], fill='#EEEEEE')
        draw.text((370, 263), "Enter email", fill='#999999')
        
        draw.rectangle([350, 310, 850, 350], fill='#EEEEEE')
        draw.text((370, 323), "Enter password", fill='#999999')
        
        draw.rectangle([350, 380, 850, 420], fill='#4267B2')
        draw.text((560, 393), "Log In", fill='#FFFFFF')
        
    elif phishing_type == 'ip_based':
        # IP-based URL (highly suspicious)
        draw.rectangle([0, 0, 1200, 50], fill='#666666')
        
        draw.rectangle([100, 60, 1100, 90], fill='#FFCCCC', outline='#FF0000')
        draw.text((120, 67), "http://192.168.1.100/secure/login.php", fill='#CC0000')
        
        # Basic HTML form
        draw.rectangle([200, 150, 1000, 650], fill='#F0F0F0', outline='#999999')
        draw.text((400, 200), "Security Verification Required", fill='#333333')
        
        # Form fields
        draw.rectangle([250, 280, 950, 320], fill='#FFFFFF', outline='#CCCCCC')
        draw.text((270, 293), "Credit Card Number:", fill='#666666')
        
        draw.rectangle([250, 340, 950, 380], fill='#FFFFFF', outline='#CCCCCC')
        draw.text((270, 353), "Expiry Date:", fill='#666666')
        
        draw.rectangle([250, 400, 950, 440], fill='#FFFFFF', outline='#CCCCCC')
        draw.text((270, 413), "CVV:", fill='#666666')
        
        draw.rectangle([250, 460, 950, 500], fill='#FFFFFF', outline='#CCCCCC')
        draw.text((270, 473), "Social Security Number:", fill='#666666')
        
        draw.rectangle([400, 530, 800, 580], fill='#FF0000')
        draw.text((540, 547), "SUBMIT", fill='#FFFFFF')
        
    else:  # prize_scam
        # Prize/lottery scam
        draw.rectangle([0, 0, 1200, 800], fill='#FFD700')
        
        draw.text((300, 100), "🎉 CONGRATULATIONS! 🎉", fill='#FF0000')
        draw.text((250, 180), "YOU HAVE WON $1,000,000!", fill='#CC0000')
        
        draw.rectangle([200, 250, 1000, 600], fill='#FFFFFF', outline='#FF0000', width=3)
        
        draw.text((300, 280), "Claim your prize NOW!", fill='#FF0000')
        draw.text((250, 340), "Enter your details to receive your winnings:", fill='#333333')
        
        # Suspicious form fields
        draw.rectangle([250, 380, 950, 420], fill='#FFFFCC', outline='#FFCC00')
        draw.text((270, 393), "Full Name:", fill='#666666')
        
        draw.rectangle([250, 440, 950, 480], fill='#FFFFCC', outline='#FFCC00')
        draw.text((270, 453), "Bank Account Number:", fill='#666666')
        
        draw.rectangle([400, 520, 800, 570], fill='#00CC00')
        draw.text((520, 537), "CLAIM NOW!", fill='#FFFFFF')
        
        draw.text((280, 620), "⏰ Offer expires in 5 MINUTES!", fill='#FF0000')
    
    img.save(path)
    print(f"  Created: {path}")

def main():
    """Generate sample screenshots for training."""
    print("=" * 60)
    print("GENERATING SAMPLE SCREENSHOTS FOR VISUAL MODEL TRAINING")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    phishing_dir = os.path.join(base_dir, 'data', 'screenshots', 'phishing')
    legitimate_dir = os.path.join(base_dir, 'data', 'screenshots', 'legitimate')
    
    os.makedirs(phishing_dir, exist_ok=True)
    os.makedirs(legitimate_dir, exist_ok=True)
    
    # Generate legitimate screenshots
    print("\n📗 Generating LEGITIMATE website screenshots...")
    brands = ['google', 'microsoft', 'amazon', 'facebook', 'linkedin', 'github']
    
    for i, brand in enumerate(brands):
        for variant in range(5):  # 5 variants each
            path = os.path.join(legitimate_dir, f"{brand}_{variant+1}.png")
            create_legitimate_screenshot(path, brand)
    
    print(f"  Total: {len(brands) * 5} legitimate screenshots")
    
    # Generate phishing screenshots
    print("\n📕 Generating PHISHING website screenshots...")
    phishing_types = ['login_form', 'typosquatting', 'ip_based', 'prize_scam']
    target_brands = ['paypal', 'google', 'microsoft', 'amazon', 'facebook', 'apple']
    
    count = 0
    for brand in target_brands:
        for ptype in phishing_types:
            for variant in range(2):
                path = os.path.join(phishing_dir, f"phish_{brand}_{ptype}_{variant+1}.png")
                create_phishing_screenshot(path, brand, ptype)
                count += 1
    
    print(f"  Total: {count} phishing screenshots")
    
    print("\n" + "=" * 60)
    print("SAMPLE SCREENSHOTS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nLocations:")
    print(f"  Legitimate: {legitimate_dir}")
    print(f"  Phishing:   {phishing_dir}")
    print(f"\nYou can now train the visual model with:")
    print(f"  py scripts/train.py --train-all")

if __name__ == "__main__":
    main()
