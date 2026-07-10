import pandas as pd
import csv
import time

def reconstruct_and_downsample(input_file, output_file, interval_us=1000000):
    """
    Parses L2 incremental updates, maintains the order book, 
    and outputs a snapshot every `interval_us` (1 second = 1,000,000 us).
    """
    bids = {} # price -> amount
    asks = {} # price -> amount
    
    print(f"Starting reconstruction from {input_file}...")
    start_time = time.time()
    
    # We will store the output rows here
    output_rows = []
    
    # Track the current interval
    current_interval = None
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount
        #   0         1       2            3              4        5     6      7
        
        count = 0
        for row in reader:
            count += 1
            if count % 5000000 == 0:
                print(f"Processed {count} ticks...")
                
            ts = int(row[2])
            side = row[5]
            price = float(row[6])
            amount = float(row[7])
            
            # Update the order book
            if side == 'bid':
                if amount == 0:
                    bids.pop(price, None)
                else:
                    bids[price] = amount
            elif side == 'ask':
                if amount == 0:
                    asks.pop(price, None)
                else:
                    asks[price] = amount
            
            # Check if we crossed into a new interval
            interval = ts // interval_us
            if current_interval is None:
                current_interval = interval
            
            if interval > current_interval:
                # We crossed a boundary, take a snapshot of the current book
                # Sort bids (descending) and asks (ascending)
                sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)
                sorted_asks = sorted(asks.items(), key=lambda x: x[0])
                
                # We need at least 5 levels to make a consistent state
                if len(sorted_bids) >= 5 and len(sorted_asks) >= 5:
                    top_bids = sorted_bids[:5]
                    top_asks = sorted_asks[:5]
                    
                    row_data = [ts]
                    # Append Top 5 Bids (Price, Amount)
                    for i in range(5):
                        row_data.extend([top_bids[i][0], top_bids[i][1]])
                    # Append Top 5 Asks (Price, Amount)
                    for i in range(5):
                        row_data.extend([top_asks[i][0], top_asks[i][1]])
                        
                    output_rows.append(row_data)
                    
                current_interval = interval

    print(f"Finished parsing in {time.time() - start_time:.2f} seconds.")
    
    # Save to CSV
    columns = ['timestamp']
    for i in range(1, 6):
        columns.extend([f'bid_price_{i}', f'bid_amount_{i}'])
    for i in range(1, 6):
        columns.extend([f'ask_price_{i}', f'ask_amount_{i}'])
        
    df = pd.DataFrame(output_rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} snapshots to {output_file}.")

if __name__ == "__main__":
    reconstruct_and_downsample(
        input_file='data/BTCUSDT_L2.csv',
        output_file='data/l2_features.csv',
        interval_us=1000000 # 1 second
    )
