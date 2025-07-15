import pandas as pd


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
def process_book_data(data, end_time=None):

    order_book = Objdict()
    order_book.Buy = Objdict()
    order_book.Sell = Objdict()
    order_book.Orders = Objdict()
    trades = []
    columns = data.columns
    if end_time:
        book_data = data.loc[data["Timestamp"] < end_time, :]
    else:
        book_data = data

    # We convert df into a list of dictionaries (one dict per row), for easier iteration and processing
    book_data = book_data.to_dict(orient="records")

    for event in book_data:

        event = Objdict(event)

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
                "Timestamp": event.Timestamp,
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
                "Timestamp": event.Timestamp,
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
                "Timestamp": event.Timestamp,
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
                "Timestamp": event.Timestamp,
                "Price": price,
                "Quantity": qty,
                "Type": "Cross"
            })

            trades.append(trade)

        else:
            print(event)

    trades = pd.DataFrame(trades)

    return order_book, trades

