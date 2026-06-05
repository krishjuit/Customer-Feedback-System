import pandas as pd
import os

# Define mock dataset of customer reviews across departments and sentiments
reviews_data = [
    # --- PRODUCT QUALITY ---
    {"review_text": "The build quality of this device is exceptional, very sturdy and durable.", "sentiment": "Positive", "department": "Product"},
    {"review_text": "Absolutely love the camera quality! Photos come out crystal clear.", "sentiment": "Positive", "department": "Product"},
    {"review_text": "This product exceeds my expectations. The screen is vibrant and battery lasts all day.", "sentiment": "Positive", "department": "Product"},
    {"review_text": "Very sleek design and extremely lightweight. High quality materials.", "sentiment": "Positive", "department": "Product"},
    {"review_text": "The processor is incredibly fast. No lag whatsoever when running multiple apps.", "sentiment": "Positive", "department": "Product"},
    {"review_text": "Highly satisfied with the product performance. It handles high-end games easily.", "sentiment": "Positive", "department": "Product"},
    
    {"review_text": "It stopped working completely after just two days of normal usage.", "sentiment": "Negative", "department": "Product"},
    {"review_text": "Very cheap plastic material. Feels like it will break easily.", "sentiment": "Negative", "department": "Product"},
    {"review_text": "The battery life is terrible, it barely lasts for three hours.", "sentiment": "Negative", "department": "Product"},
    {"review_text": "Screen has flickering issues right out of the box. Very disappointed.", "sentiment": "Negative", "department": "Product"},
    {"review_text": "The software is full of bugs and crashes constantly. Do not recommend.", "sentiment": "Negative", "department": "Product"},
    {"review_text": "The device gets extremely hot after only ten minutes of basic web browsing.", "sentiment": "Negative", "department": "Product"},
    
    {"review_text": "The product quality is average. Nothing outstanding, but it works.", "sentiment": "Neutral", "department": "Product"},
    {"review_text": "Decent build quality, though the screen could be a bit brighter.", "sentiment": "Neutral", "department": "Product"},
    {"review_text": "It is a basic model with standard features. It does the job.", "sentiment": "Neutral", "department": "Product"},
    {"review_text": "Performance is acceptable for the price. Not fast, but not too slow.", "sentiment": "Neutral", "department": "Product"},

    # --- SHIPPING AND DELIVERY ---
    {"review_text": "Super fast delivery! Ordered yesterday and it arrived today.", "sentiment": "Positive", "department": "Shipping"},
    {"review_text": "The packaging was very secure, double boxed and well protected.", "sentiment": "Positive", "department": "Shipping"},
    {"review_text": "Received the item earlier than estimated. Great shipping service.", "sentiment": "Positive", "department": "Shipping"},
    {"review_text": "Excellent shipping tracking, knew exactly when it would arrive.", "sentiment": "Positive", "department": "Shipping"},
    {"review_text": "Item arrived in perfect condition, packaging was top-notch.", "sentiment": "Positive", "department": "Shipping"},
    {"review_text": "Extremely fast transit times. Delivery driver was also very polite.", "sentiment": "Positive", "department": "Shipping"},
    
    {"review_text": "The package arrived two weeks late. Extremely slow delivery.", "sentiment": "Negative", "department": "Shipping"},
    {"review_text": "The box was completely crushed and damaged when the courier delivered it.", "sentiment": "Negative", "department": "Shipping"},
    {"review_text": "They delivered the item to the wrong house! Had to go find it myself.", "sentiment": "Negative", "department": "Shipping"},
    {"review_text": "Zero updates on my shipment. Tracking number did not work at all.", "sentiment": "Negative", "department": "Shipping"},
    {"review_text": "Package was lost in transit and delivery was delayed indefinitely.", "sentiment": "Negative", "department": "Shipping"},
    {"review_text": "The delivery was delayed three times without any explanation from the courier.", "sentiment": "Negative", "department": "Shipping"},
    
    {"review_text": "Delivery took about a week, which is normal for standard free shipping.", "sentiment": "Neutral", "department": "Shipping"},
    {"review_text": "The packaging was standard. The box had minor scuffs but the item inside was fine.", "sentiment": "Neutral", "department": "Shipping"},
    {"review_text": "Arrived on the scheduled day. Tracking updates were a bit delayed but worked.", "sentiment": "Neutral", "department": "Shipping"},
    {"review_text": "Average delivery experience. Nothing special to note.", "sentiment": "Neutral", "department": "Shipping"},

    # --- CUSTOMER SUPPORT ---
    {"review_text": "The customer service agent was incredibly helpful and solved my issue in minutes.", "sentiment": "Positive", "department": "Support"},
    {"review_text": "Great support! They immediately processed my refund with no questions asked.", "sentiment": "Positive", "department": "Support"},
    {"review_text": "The live chat assistant was very polite and professional.", "sentiment": "Positive", "department": "Support"},
    {"review_text": "Received a prompt response from the support team within 10 minutes.", "sentiment": "Positive", "department": "Support"},
    {"review_text": "Very satisfied with how support handled my warranty claim. Smooth process.", "sentiment": "Positive", "department": "Support"},
    {"review_text": "They went above and beyond to help me set up my new device.", "sentiment": "Positive", "department": "Support"},
    
    {"review_text": "Horrible customer support! I asked for assistance and was completely ignored.", "sentiment": "Negative", "department": "Support"},
    {"review_text": "Waited on hold for over an hour and then the call got disconnected.", "sentiment": "Negative", "department": "Support"},
    {"review_text": "The support representative was very rude and unhelpful.", "sentiment": "Negative", "department": "Support"},
    {"review_text": "Sent three emails to support and received absolutely no response.", "sentiment": "Negative", "department": "Support"},
    {"review_text": "They refused to help me with my damaged item. Terrible service.", "sentiment": "Negative", "department": "Support"},
    {"review_text": "Getting a refund was a complete nightmare. Support kept transferring my call.", "sentiment": "Negative", "department": "Support"},
    
    {"review_text": "The support agent answered my question, but took a while to connect.", "sentiment": "Neutral", "department": "Support"},
    {"review_text": "Response from customer service was average. Got a standard template answer.", "sentiment": "Neutral", "department": "Support"},
    {"review_text": "They resolved the issue, but the process was somewhat bureaucratic.", "sentiment": "Neutral", "department": "Support"},
    {"review_text": "The chatbot was okay, but I eventually had to speak with a human agent anyway.", "sentiment": "Neutral", "department": "Support"},

    # --- PRICE AND VALUE ---
    {"review_text": "Excellent value for money. Much cheaper than other brands but performs just as well.", "sentiment": "Positive", "department": "Price"},
    {"review_text": "Got it on discount, absolute steal for this price!", "sentiment": "Positive", "department": "Price"},
    {"review_text": "Very reasonably priced and definitely worth every penny.", "sentiment": "Positive", "department": "Price"},
    {"review_text": "Affordable option that does not compromise on quality or performance.", "sentiment": "Positive", "department": "Price"},
    {"review_text": "A great investment. Highly competitive pricing for these features.", "sentiment": "Positive", "department": "Price"},
    {"review_text": "You get high-end features at a mid-range price tag. Fantastic value.", "sentiment": "Positive", "department": "Price"},
    
    {"review_text": "Way too expensive for what it offers. Definitely overpriced.", "sentiment": "Negative", "department": "Price"},
    {"review_text": "Not worth the money at all. You can get better models for half the price.", "sentiment": "Negative", "department": "Price"},
    {"review_text": "Very disappointed. It is overpriced and feels low quality.", "sentiment": "Negative", "department": "Price"},
    {"review_text": "The pricing is ridiculous. They are charging double for basic features.", "sentiment": "Negative", "department": "Price"},
    {"review_text": "Total waste of money. Completely overpriced and underperforming.", "sentiment": "Negative", "department": "Price"},
    {"review_text": "Premium price tag for budget-level hardware. Absolute rip-off.", "sentiment": "Negative", "department": "Price"},
    
    {"review_text": "The price is fair, matches the market average for this category.", "sentiment": "Neutral", "department": "Price"},
    {"review_text": "Not cheap, but not too expensive either. Price is okay.", "sentiment": "Neutral", "department": "Price"},
    {"review_text": "It is priced appropriately for what you get. Fair deal.", "sentiment": "Neutral", "department": "Price"},
    {"review_text": "Standard pricing. Worth it if you buy it during sales.", "sentiment": "Neutral", "department": "Price"},
    
    # --- GENERAL MIXED FEEDBACK (For extra volume and variance) ---
    {"review_text": "Great product but the shipping took slightly longer than expected.", "sentiment": "Positive", "department": "Product"},
    {"review_text": "Excellent customer service, but the delivery package was wet.", "sentiment": "Positive", "department": "Support"},
    {"review_text": "The delivery was fast, but the price is too high for this item.", "sentiment": "Neutral", "department": "Price"},
    {"review_text": "Support helped me, but the product itself is quite mediocre.", "sentiment": "Neutral", "department": "Product"},
    {"review_text": "The box was damaged but the support agent replaced it immediately. Superb service!", "sentiment": "Positive", "department": "Support"},
    {"review_text": "Too expensive and it arrived broken. Returning it immediately.", "sentiment": "Negative", "department": "Price"},
    {"review_text": "Decent price, but it took three weeks to arrive and support did not care.", "sentiment": "Negative", "department": "Shipping"},
    {"review_text": "It works as advertised. The pricing is okay and delivery was standard.", "sentiment": "Neutral", "department": "Product"},
    {"review_text": "Outstanding build quality and fast shipping! Totally satisfied.", "sentiment": "Positive", "department": "Product"},
    {"review_text": "Very poor performance, bad customer service, and overpriced.", "sentiment": "Negative", "department": "Product"},
    {"review_text": "I got what I ordered. Delivery was prompt. No complaints.", "sentiment": "Positive", "department": "Shipping"},
    {"review_text": "The support team was helpful, but they could not fix my broken hardware.", "sentiment": "Neutral", "department": "Support"},
    {"review_text": "It is okay. Price matches quality. Delivery was fine.", "sentiment": "Neutral", "department": "Price"},
    {"review_text": "The delivery was delayed but the product is amazing. So happy!", "sentiment": "Positive", "department": "Product"},
    {"review_text": "Support took forever to reply. The product is also quite cheap-feeling.", "sentiment": "Negative", "department": "Support"},
    {"review_text": "The price is competitive, but shipping costs are too high.", "sentiment": "Neutral", "department": "Price"}
]

def generate_dataset():
    df = pd.DataFrame(reviews_data)
    output_path = os.path.join(os.path.dirname(__file__), "feedback_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated dataset with {len(df)} rows at: {output_path}")

if __name__ == "__main__":
    generate_dataset()
