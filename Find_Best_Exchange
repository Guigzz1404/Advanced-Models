import pandas as pd
import datetime
from datetime import timedelta


# Display every lines
pd.set_option('display.max_rows', None)


def get_top_of_book(ob):
    from collections import defaultdict

    top_of_book = defaultdict(lambda: {"BestBid": -float('inf'), "BestAsk": -float('inf')})

    # For buy orders
    for o_id, order in ob.Buy.items():
        ex = order.Exchange
        price = float(order.Price)/10000
        if top_of_book[ex]["BestBid"] is None or price > top_of_book[ex]["BestBid"]:
            top_of_book[ex]["BestBid"] = price

    # For sell orders
    for o_id, order in ob.Sell.items():
        ex = order.Exchange
        price = float(order.Price)/10000
        if top_of_book[ex]["BestAsk"] is None or price < top_of_book[ex]["BestAsk"]:
            top_of_book[ex]["BestAsk"] = price

    return dict(top_of_book)

# Object creation which acts like a dictionary
class Objdict(dict):

    # Redirects attribute access (e.g. obj.foo) to dictionary-style lookup (self['foo']) if the key exists
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    # Redirects attribute assignments (e.g. obj.foo = val) to dictionary-style storage (self['foo'] = val)
    def __setattr__(self, name, value):
        self[name] = value

    # Redirects attribute deletion (e.g. del obj.foo) to dictionary-style deletion (del self['foo'])
    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


# end_time is an optional parameter which allow to filter data in function of the time
def process_book_data(data, start_time=None, end_time=None):

    order_book = Objdict()
    order_book.Buy = Objdict()
    order_book.Sell = Objdict()
    order_book.Orders = Objdict()
    trades = []
    top_list = []
    if end_time:
        book_data = data.loc[data["Timestamp"] < end_time, :]
    else:
        book_data = data

    # We convert df into a list of dictionaries (one dict per row), for easier iteration and processing
    book_data = book_data.to_dict(orient="records")

    # Next trigger to look at the best bid
    next_trigger_time = start_time

    for i, event in enumerate(book_data):

        event = Objdict(event)
        current_ts = event.Timestamp

        # Creation of a unique number by concat of orderNumber and Exchange ticker
        o_num_unique = f'{event.OrderNumber}_{event.Exchange}'  # OrderNumber only unique on a specific exchange

        if event.Event == 'B':  # Buy Order
            order_book.Orders[o_num_unique] = 'Buy'
            order_book.Buy[o_num_unique] = event

        elif event.Event == 'S':  # Sell Order
            order_book.Orders[o_num_unique] = 'Sell'
            order_book.Sell[o_num_unique] = event

        elif event.Event == 'D': # Delete Order
            side = order_book.Orders[o_num_unique] # We get the side of the order to remove order_book[side]
            del order_book.Orders[o_num_unique]
            del order_book[side][o_num_unique]

        elif event.Event == 'C': # Partial Cancellation
            side = order_book.Orders[o_num_unique]
            order = order_book[side][o_num_unique]
            qty = int(event.Quantity)

            order.Quantity -= qty # We subtract quantity to cancel from initial quantity
            if order.Quantity <= 0:
                del order_book.Orders[o_num_unique]
                del order_book[side][o_num_unique]
            else:
                order_book[side][o_num_unique] = order

        elif event.Event == 'T': # Dark Execution
            qty = int(event.Quantity)
            price = float(event.Price)/10000
            trade = Objdict({
                "Timestamp": current_ts,
                "Price": price,
                "Quantity": qty,
                "Type": "Dark"
            })

            trades.append(trade)

        elif event.Event == 'F': # Lit Full Execution
            side = order_book.Orders[o_num_unique]
            order = order_book[side][o_num_unique]
            qty = int(order.Quantity) # We use order instead of event in case where some modifications have been added (C)
            price = float(order.Price)/10000

            trade = Objdict({
                "Timestamp": current_ts,
                "Price": price,
                "Quantity": qty,
                "Type": "Lit"
            })

            trades.append(trade)

            del order_book.Orders[o_num_unique]
            del order_book[side][o_num_unique]

        elif event.Event == 'E': # Lit Partial Execution
            side = order_book.Orders[o_num_unique]
            order = order_book[side][o_num_unique]
            qty = int(event.Quantity)
            price = float(order.Price)/10000

            trade = Objdict({
                "Timestamp": current_ts,
                "Price": price,
                "Quantity": qty,
                "Type": "Lit"
            })

            trades.append(trade)

            order.Quantity -= qty
            if order.Quantity <= 0:
                del order_book.Orders[o_num_unique]
                del order_book[side][o_num_unique]
            else:
                order_book[side][o_num_unique] = order

        elif event.Event == 'X': # Auction Cross
            qty = int(event.Quantity)
            price = float(event.Price)/10000
            trade = Objdict({
                "Timestamp": current_ts,
                "Price": price,
                "Quantity": qty,
                "Type": "Cross"
            })

            trades.append(trade)

        else:
            print(event)

        # Find when it's a 5-min multiple from beginning of the trading session
        if current_ts >= next_trigger_time:
            top = get_top_of_book(order_book)
            top_list.append(top)
            next_trigger_time += timedelta(minutes=5)

    trades = pd.DataFrame(trades)

    return order_book, trades, top_list


# DATA TREATMENT
# Read the LevelIII data
csco_data = pd.read_csv("csco_levelIII_data.csv")
dt = datetime.date(2025,7, 9)
# Convert Timestamp column into datetime object
csco_data["Timestamp"] = pd.to_datetime(csco_data["Timestamp"], unit="ms", origin=dt)
csco_data = csco_data.drop(columns=["Ticker", "MPID"])

# Process the LevelIII Data and get top of the book per exchange
order_book, trades, top_list = process_book_data(csco_data, pd.Timestamp("2025-07-09 09:30:00.000"), pd.Timestamp("2025-07-09 16:00:00.000"))

# Process the top order list per exchange to find the best exchange
top_exchange = []
for snapshot in top_list:
    bids = {ex: info["BestBid"] for ex, info in snapshot.items()}
    asks = {ex: info['BestAsk'] for ex, info in snapshot.items()}

    # Find Best values
    best_bid_val = max(bids.values())
    best_ask_val = min(asks.values())

    # Find if best bid/ask is proposed by one exchange or several
    best_bid_ex = [ex for ex, val in bids.items() if val == best_bid_val]
    best_ask_ex = [ex for ex, val in asks.items() if val == best_ask_val]

    # If several, return NaN
    bid_result = best_bid_ex[0] if len(best_bid_ex) == 1 else float('nan')
    ask_result = best_ask_ex[0] if len(best_ask_ex) == 1 else float('nan')

    top_exchange.append({
        "BestBidExchange": bid_result,
        "BestAskExchange": ask_result
    })

top_exchange = pd.DataFrame(top_exchange)
modes = top_exchange.mode().iloc[0]
print(modes)
