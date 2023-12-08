import pandas as pd
import numpy as np
import os
import gzip
from collections.abc import Callable
from datetime import datetime
from sklearn.linear_model import LinearRegression

DATA_DIR = os.environ['DATA_DIRECTORY']

class BarTransformer:
    """
    Class to convert raw trades to different style sampled OHLC bars.
    """
    def __init__(self, chunk_size: int=0):
        self.chunk_size = chunk_size
    
    
    # =============================================================================
    # Time bars
        
    def time_process_df(self, chunk: pd.DataFrame, sample_time: str, initial_state: dict=None) -> tuple:
        """
        Process a chunk of trade data and convert it into time-based OHLC (Open, High, Low, Close) bars.

        Parameters
        ----------
        chunk : pd.DataFrame
            The chunk of trade data to be processed.
        sample_time : str
            The time interval for resampling data into OHLC bars (e.g., '1H' for hourly bars).
        initial_state : dict, optional
            Initial state information for processing the chunk (e.g., cumulative volume, prices).
            Ignored in this function, argument in function for consistency on loading functions.
        
        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing two elements:
            - pd.DataFrame: DataFrame containing trade-based OHLC bars.
            - dict: The final state information after processing the chunk.

        Raises
        ------
        KeyError
            If the 'datetime' or 'timestamp' column is not present in the input chunk.

        Notes
        -----
        This function processes trade data by resampling it into time-based OHLC bars, including
        calculating open, high, low, close prices, average price, trade count, and volume.

        Example
        -------
        chunk = pd.read_csv('sample_trade_data.csv')
        sample_time = '1H'
        ohlc_bars = self.time_process_df(chunk, sample_time)
        """
        
        # Check for 'datetime' column since pandas uses those to resample.
        # If not present attempts to use 'timestamp' column or else raises KeyError.
        if 'datetime' not in chunk.columns:
            try:
                chunk['datetime'] = chunk['timestamp'].apply(datetime.fromtimestamp)
            except KeyError:
                raise KeyError("'timestamp' column not present in trades df chunk. No column to create time column.")
        
        chunk = chunk.set_index(['datetime'])
        groups = chunk.resample(sample_time)
        
        ohlcs = groups['price'].ohlc()
        avgs = groups['price'].mean()
        volumes = groups['size'].sum()
        counts = groups['price'].count()

        ohlc_bars = pd.concat([ohlcs, avgs, volumes, counts], axis=1)
        ohlc_bars = ohlc_bars.reset_index(drop=False)
        ohlc_bars.columns = ['datetime','open','high','low','close','avg','volume','trades']
        
        # Returning the intial state (None) for consistencyy with other processing funcs.
        final_state = initial_state
        
        return ohlc_bars.reset_index(drop=True), final_state
        

    

    # =============================================================================
    # Trade (tick) bars
    
    def trades_process_df(self, chunk: pd.DataFrame, trades_threshold: int, initial_state: dict=None) -> tuple:
        """
        Process a chunk of trade data and convert it into trade-based OHLC (Open, High, Low, Close) bars.

        Parameters
        ----------
        chunk : pd.DataFrame
            The chunk of trade data to be processed.
        trades_threshold : int
            The number of trades required to form a single OHLC bar.
        initial_state : dict, optional
            Initial state information for processing the chunk (e.g., cumulative volume, prices).
        
        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing two elements:
            - pd.DataFrame: DataFrame containing trade-based OHLC bars.
            - dict: The final state information after processing the chunk.

        Notes
        -----
        This function processes trade data by aggregating it into trade-based OHLC bars. It calculates open,
        high, low, close prices, and VWAP (Volume-Weighted Average Price) for each bar.

        Example
        -------
        chunk = pd.read_csv('sample_trade_data.csv')
        trades_threshold = 100
        ohlc_bars, final_state = self.trades_process_df(chunk, trades_threshold)
        """
        cumulative_volume = initial_state['cumulative_volume'] if initial_state else 0
        cumulative_price = initial_state['cumulative_price'] if initial_state else 0
        cumulative_trades = initial_state['cumulative_trades'] if initial_state else 0
        start_at = initial_state['start_at'] if initial_state else None
        open_price = initial_state['open_price'] if initial_state else None
        high_price = initial_state['high_price'] if initial_state else None
        low_price = initial_state['low_price'] if initial_state else None
        close_price = initial_state['close_price'] if initial_state else None
        
        # Create an empty DataFrame to store OHLC bars and VWAP    
        ohlc_bars = pd.DataFrame(columns=['start_at','end_at','open','high','low','close','vwap','trades'])
        
        # Iterate through the trade data in the current chunk
        for index,trade in enumerate(chunk.to_dict('records')):
            # Update cumulative volume and cumulative price
            cumulative_volume += trade['size']
            cumulative_price += trade['price'] * trade['size']
            cumulative_trades += 1
            
            # Update OHLC values
            if open_price is None:
                open_price = trade['price']
                high_price = trade['price']
                low_price = trade['price']
                start_at = trade['timestamp']
            else:
                high_price = max(high_price, trade['price'])
                low_price = min(low_price, trade['price'])
            close_price = trade['price']

            # Check if the cumulative volume threshold is reached
            if cumulative_trades >= trades_threshold:
                # Calculate VWAP for the bar
                vwap = cumulative_price / cumulative_volume

                # Append OHLC bar and VWAP to the OHLC bars DataFrame
                temp_ohlc = pd.DataFrame({'start_at': start_at,
                                          'end_at': trade['timestamp'],
                                          'open': open_price,
                                          'high': high_price,
                                          'low': low_price,
                                          'close': close_price,
                                          'vwap': vwap,
                                          'volume': cumulative_volume,
                                          'trades': cumulative_trades},
                                         index=[0])
                
                ohlc_bars = pd.concat([ohlc_bars, temp_ohlc], axis=0)

                # Reset cumulative volume, cumulative price, and OHLC values
                cumulative_volume = 0
                cumulative_price = 0
                cumulative_trades = 0
                open_price = None
                high_price = None
                low_price = None
                close_price = None

        final_state = {
                'cumulative_volume': cumulative_volume,
                'cumulative_price': cumulative_price,
                'cumulative_trades': cumulative_trades,
                'start_at': start_at,
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'close_price': close_price,
                }
        
        return ohlc_bars.reset_index(drop=True), final_state
    
    

    
    # =============================================================================
    # Dollar bars
    
    def dollar_process_df(self, chunk: pd.DataFrame, dollar_threshold: int, initial_state: dict=None) -> tuple:
        cumulative_volume = initial_state['cumulative_volume'] if initial_state else 0
        cumulative_price = initial_state['cumulative_price'] if initial_state else 0
        cumulative_dollar = initial_state['cumulative_dollar'] if initial_state else 0
        start_at = initial_state['start_at'] if initial_state else None
        open_price = initial_state['open_price'] if initial_state else None
        high_price = initial_state['high_price'] if initial_state else None
        low_price = initial_state['low_price'] if initial_state else None
        close_price = initial_state['close_price'] if initial_state else None
        n_trades = initial_state['n_trades'] if initial_state else 0
    
        # Create an empty DataFrame to store OHLC bars and VWAP    
        ohlc_bars = pd.DataFrame(columns=['start_at','end_at','open','high','low','close','vwap','volume','trades','dollar_volume'])
        
        # Iterate through the trade data in the current chunk
        for index,trade in enumerate(chunk.to_dict('records')):
            # Update cumulative volume and cumulative price
            cumulative_volume += trade['size']
            cumulative_price += trade['price'] * trade['size']
            n_trades += 1
            cumulative_dollar += trade['foreignNotional']
        
            # Update OHLC values
            if open_price is None:
                open_price = trade['price']
                high_price = trade['price']
                low_price = trade['price']
                start_at = trade['timestamp']
            else:
                high_price = max(high_price, trade['price'])
                low_price = min(low_price, trade['price'])
            close_price = trade['price']
            
            # Check if the cumulative dollar threshold is reached
            if cumulative_dollar >= dollar_threshold:
                # Calculate VWAP for the bar
                vwap = cumulative_price / cumulative_volume

                # Append OHLC bar and VWAP to the OHLC bars DataFrame
                temp_ohlc = pd.DataFrame({'start_at': start_at,
                                          'end_at': trade['timestamp'],
                                          'open': open_price,
                                          'high': high_price,
                                          'low': low_price,
                                          'close': close_price,
                                          'vwap': vwap,
                                          'volume': cumulative_volume,
                                          'trades':n_trades,
                                          'dollar_volume': cumulative_dollar},
                                         index=[0])
                
                ohlc_bars = pd.concat([ohlc_bars, temp_ohlc], axis=0)

                # Reset cumulative volume, cumulative price, and OHLC values
                cumulative_volume = 0
                cumulative_price = 0
                cumulative_dollar = 0                
                start_at = None
                open_price = None
                high_price = None
                low_price = None
                close_price = None
                n_trades = 0

        final_state = {
                'cumulative_volume': cumulative_volume,
                'cumulative_price': cumulative_price,
                'cumulative_dollar': cumulative_dollar,
                'start_at': start_at,
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'close_price': close_price,
                'n_trades': n_trades,
                }
        
        return ohlc_bars.reset_index(drop=True), final_state
        
            
            

    # =============================================================================
    # Volume bars
    
    def volume_process_df(self, chunk: pd.DataFrame, volume_threshold: int, initial_state: dict=None) -> tuple:
        """
        Process a chunk of trade data and convert it into volume-based OHLC (Open, High, Low, Close) bars.

        Parameters
        ----------
        chunk : pd.DataFrame
            The chunk of trade data to be processed.
        volume_threshold : int
            The cumulative trade volume required to form a single OHLC bar.
        initial_state : dict, optional
            Initial state information for processing the chunk (e.g., cumulative volume, prices).

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing two elements:
            - pd.DataFrame: A DataFrame containing volume-based OHLC bars.
            - dict: The final state information after processing the chunk.

        Notes
        -----
        This function processes trade data by aggregating it into volume-based OHLC bars. It calculates open,
        high, low, close prices, VWAP (Volume-Weighted Average Price), and the number of trades for each bar.

        Example
        -------
        chunk = pd.read_csv('sample_trade_data.csv')
        volume_threshold = 1000
        ohlc_bars, final_state = self.volume_process_df(chunk, volume_threshold)
        """
        cumulative_volume = initial_state['cumulative_volume'] if initial_state else 0
        cumulative_price = initial_state['cumulative_price'] if initial_state else 0
        start_at = initial_state['start_at'] if initial_state else None
        open_price = initial_state['open_price'] if initial_state else None
        high_price = initial_state['high_price'] if initial_state else None
        low_price = initial_state['low_price'] if initial_state else None
        close_price = initial_state['close_price'] if initial_state else None
        n_trades = initial_state['n_trades'] if initial_state else 0
    
        # Create an empty DataFrame to store OHLC bars and VWAP    
        ohlc_bars = pd.DataFrame(columns=['start_at','end_at','open','high','low','close','vwap','volume','trades'])
        
        # Iterate through the trade data in the current chunk
        for index,trade in enumerate(chunk.to_dict('records')):
            # Update cumulative volume and cumulative price
            cumulative_volume += trade['size']
            cumulative_price += trade['price'] * trade['size']
            n_trades += 1
            
            # Update OHLC values
            if open_price is None:
                open_price = trade['price']
                high_price = trade['price']
                low_price = trade['price']
                start_at = trade['timestamp']
            else:
                high_price = max(high_price, trade['price'])
                low_price = min(low_price, trade['price'])
            close_price = trade['price']

            # Check if the cumulative volume threshold is reached
            if cumulative_volume >= volume_threshold:
                # Calculate VWAP for the bar
                vwap = cumulative_price / cumulative_volume

                # Append OHLC bar and VWAP to the OHLC bars DataFrame
                temp_ohlc = pd.DataFrame({'start_at': start_at,
                                          'end_at': trade['timestamp'],
                                          'open': open_price,
                                          'high': high_price,
                                          'low': low_price,
                                          'close': close_price,
                                          'vwap': vwap,
                                          'volume': cumulative_volume,
                                          'trades':n_trades},
                                         index=[0])
                
                ohlc_bars = pd.concat([ohlc_bars, temp_ohlc], axis=0)

                # Reset cumulative volume, cumulative price, and OHLC values
                cumulative_volume = 0
                cumulative_price = 0
                start_at = None
                open_price = None
                high_price = None
                low_price = None
                close_price = None
                n_trades = 0

        final_state = {
                'cumulative_volume': cumulative_volume,
                'cumulative_price': cumulative_price,
                'start_at': start_at,
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'close_price': close_price,
                'n_trades': n_trades,
                }
        
        return ohlc_bars.reset_index(drop=True), final_state
    
    
    
    
    # =============================================================================
    # Volatility bars
    
    def volatility_process_df(self, chunk: pd.DataFrame, volatility_threshold: int, initial_state: dict=None) -> tuple:
        """
        Process a chunk of trade data and convert it into volume-based OHLC (Open, High, Low, Close) bars.

        Parameters
        ----------
        chunk : pd.DataFrame
            The chunk of trade data to be processed.
        volume_threshold : int
            The cumulative trade volume required to form a single OHLC bar.
        initial_state : dict, optional
            Initial state information for processing the chunk (e.g., cumulative volume, prices).

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing two elements:
            - pd.DataFrame: A DataFrame containing volume-based OHLC bars.
            - dict: The final state information after processing the chunk.

        Notes
        -----
        This function processes trade data by aggregating it into volatility-based OHLC bars. It calculates open,
        high, low, close prices, VWAP (Volume-Weighted Average Price), number of trades, and volatility for each bar.

        Example
        -------
        chunk = pd.read_csv('sample_trade_data.csv')
        volume_threshold = 1000
        ohlc_bars, final_state = self.volatility_process_df(chunk, volume_threshold)
        """
        cumulative_volume = initial_state['cumulative_volume'] if initial_state else 0
        cumulative_price = initial_state['cumulative_price'] if initial_state else 0
        cumulative_volatility = initial_state['cumulative_volatility'] if initial_state else 0
        prev_price = initial_state['prev_price'] if initial_state else 0
        start_at = initial_state['start_at'] if initial_state else None
        open_price = initial_state['open_price'] if initial_state else None
        high_price = initial_state['high_price'] if initial_state else None
        low_price = initial_state['low_price'] if initial_state else None
        close_price = initial_state['close_price'] if initial_state else None
        n_trades = initial_state['n_trades'] if initial_state else 0
    
        # Create an empty DataFrame to store OHLC bars and VWAP    
        ohlc_bars = pd.DataFrame(columns=['start_at','end_at','open','high','low','close','vwap','volume','trades','volatility'])
        
        # Iterate through the trade data in the current chunk
        for index,trade in enumerate(chunk.to_dict('records')):
            # Update cumulative volume and cumulative price
            cumulative_volume += trade['size']
            cumulative_price += trade['price'] * trade['size']
            n_trades += 1
            
            # Calculate the current trade (t -> t-1) variance value
            # If first iterration (prev_price is 0) then set rv to 0.
            # see: https://quant.stackexchange.com/questions/70282/how-to-determine-which-realized-volatility-estimator-should-be-used
            if prev_price == 0:
                realized_variance = 0
            else:
                realized_variance = abs((trade['price'] - prev_price) / prev_price)
                
            cumulative_volatility += realized_variance
            
            # Update OHLC values
            if open_price is None:
                open_price = trade['price']
                high_price = trade['price']
                low_price = trade['price']
                start_at = trade['timestamp']
            else:
                high_price = max(high_price, trade['price'])
                low_price = min(low_price, trade['price'])
            close_price = trade['price']
            
            # Check if the cumulative volatility threshold is reached
            if cumulative_volatility >= volatility_threshold:
                # Calculate VWAP for the bar
                vwap = cumulative_price / cumulative_volume

                # Append OHLC bar and VWAP to the OHLC bars DataFrame
                temp_ohlc = pd.DataFrame({'start_at': start_at,
                                          'end_at': trade['timestamp'],
                                          'open': open_price,
                                          'high': high_price,
                                          'low': low_price,
                                          'close': close_price,
                                          'vwap': vwap,
                                          'volume': cumulative_volume,
                                          'trades':n_trades,
                                          'volatility': cumulative_volatility},
                                         index=[0])
                
                ohlc_bars = pd.concat([ohlc_bars, temp_ohlc], axis=0)

                # Reset cumulative volume, cumulative price, and OHLC values
                cumulative_volume = 0
                cumulative_price = 0
                cumulative_volatility = 0
                start_at = None
                open_price = None
                high_price = None
                low_price = None
                close_price = None
                n_trades = 0
            
            # Setting previous price variable to use for the change between trades.
            prev_price = trade['price']

        final_state = {
                'cumulative_volume': cumulative_volume,
                'cumulative_price': cumulative_price,
                'cumulative_volatility': cumulative_volatility,
                'prev_price': prev_price,
                'start_at': start_at,
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'close_price': close_price,
                'n_trades': n_trades,
                }
        
        return ohlc_bars.reset_index(drop=True), final_state
        
    
    
    
    # =============================================================================
    # Load and transform functions. Regular at top, dynamic and helper funcs below.
            
    def _select_processing_function(self, method: str) -> Callable:
        if method == 'time':
            return self.time_process_df
        elif method == 'trades':
            return self.trades_process_df
        elif method == 'dollar':
            return self.dollar_process_df
        elif method == 'volume':
            return self.volume_process_df
        elif method == 'volatility':
            return self.volatility_process_df
        elif method == 'tick_imbalance':
            return self.tick_imbalance_process_df
    


    def daily_value_sum(self, chunk: pd.DataFrame, method: str) -> float:
        """
        Selects and calculates the chunk value based on the given processing method used.
        ie the volume processing method will use 'size', trades use len(chunk), 
        dollar use 'foreignNotional', etc.
        """
        if method in ('time', 'trades'):
            # 'time' doesnt really matter since it needs no daily value warmups.
            daily_value = len(chunk)
        elif method == 'dollar':
            daily_value = chunk['foreignNotional'].sum()
        elif method == 'volume':
            daily_value = chunk['size'].sum()
        elif method == 'volatility':
            returns_abs = abs(chunk['price'].pct_change())
            daily_value = returns_abs.sum()
        # Not sure yet.
        # Will probably be that abs_init_expected_imbal.
        #elif method == 'tick_imbalance':
        #    return self.tick_imbalance_process_df
        return daily_value
        

        

    def load_transform_ohlc_bars(self, symbol: str, method: str, value_threshold: int, **kwargs) -> pd.DataFrame:
        date_range = kwargs.get("date_range")
        print_files = kwargs.get("print_files", False)
        compressed = kwargs.get("compressed", False)
        symbol_csvs = os.listdir(os.path.join(DATA_DIR, symbol))
        initial_state = None
        # Initialize the result DataFrame
        result_ohlc_bars = pd.DataFrame()
        # List to store daily selected values.
        self.daily_values = []
        # Counter var to check against lookback_window.
        self.i = 0
        # Select function to process file chunks.
        self.process_function = self._select_processing_function(method)
    
        for csv in symbol_csvs:
            if date_range:
                if csv.split(".")[0][-10:] not in date_range:
                    continue
            # Set file_path and get the volume ohlc bars.
            file_path = os.path.join(DATA_DIR, symbol, csv)
            self.i += 1
            
            # Select the file opening type based on kwarg.
            if compressed:
                file_context = gzip.open(file_path, 'rt')
            else:
                file_context = open(file_path, 'r')
            
            # Open file and read based on lookback and chunk size.
            with file_context as file:
                if self.chunk_size == 0:
                    chunk = pd.read_csv(file)
                    ohlc_bars, final_state = self.process_function(chunk, value_threshold, initial_state)
                    self.daily_values.append(self.daily_value_sum(chunk, method))
                else:
                    # Read the csv as chunks and process each chunk on its own, concatting into total for file.
                    chunks_volume = 0
                    ohlc_chunk_bars = pd.DataFrame()
                    for chunk in pd.read_csv(file, chunksize=self.chunk_size):
                        # Temporarily add each chunk volume to get the total for day CSV.
                        chunks_volume += self.daily_value_sum(chunk, method)
                        ohlc_bars, final_state = self.process_function(chunk, value_threshold, initial_state)
                        ohlc_chunk_bars = pd.concat([ohlc_chunk_bars, ohlc_bars], axis=0)
                    self.daily_values.append(chunks_volume)
                    ohlc_bars = ohlc_chunk_bars
                
                # Concatenate the resulting OHLC bars DataFrame with the overall result
                result_ohlc_bars = pd.concat([result_ohlc_bars, ohlc_bars], ignore_index=True)
                # Update the initial state for the next file
                initial_state = final_state
                if print_files:
                    print(file_path)
    
        return result_ohlc_bars



        
    def _calculate_lookback_value(self, lookback_window: int, lookback_type: str) -> float:
        """
        Method to calculate the lookback value to use for the current day bars based on
        the class daily_values attribute.
        """
        if lookback_type in ('avg','average','mean'):
            return np.mean(self.daily_values[self.i-lookback_window:self.i])
        elif lookback_type == 'median':
            return np.median(self.daily_values[self.i-lookback_window:self.i])
        elif lookback_type == 'ewm':
            return pd.Series(self.daily_values).ewm(span=lookback_window).mean().iloc[-1]
        elif lookback_type == 'regression':
            linreg = LinearRegression(fit_intercept=False)
            # Last lookback days -1 as X and y as the last current value to be fit on.
            # Adjustments needed since 'i' counter added before this called in function.
            X = np.array(self.daily_values[self.i-lookback_window-1:self.i-2]).reshape(1,-1)
            y = np.array(self.daily_values[-1]).reshape(1,-1)
            linreg.fit(X, y)
            # Predict using the last lookback days period to predict what would be next days values.
            pred = linreg.predict(
                    np.array(self.daily_values[self.i-lookback_window:self.i]).reshape(1,-1)
                    )
            return pred




    def dynamic_load_transform_ohlc_bars(self, symbol: str, method: str, lookback_window: int, split_factor: int, **kwargs) -> pd.DataFrame:
        """
        Load and transform trade data for a given symbol into dynamically calculated OHLC bars using various methods.

        Parameters
        ----------
        symbol : str
            The trading symbol for which to load and transform trade data.
        method : str
            The method used to calculate OHLC bars. 
            Supported methods: 'trades', 'dollar', 'volume', 'volatility', 'tick_imbalance'.
        lookback_window : int
            The number of days to look back for calculating average daily values.
        split_factor : int
            A factor to divide the average daily value for calculating the value threshold.
        **kwargs : keyword arguments, optional
            Additional options for loading and processing trade data.
            - date_range : list, optional
                A list of date ranges to filter the trade data.
            - lookback_type : str, optional
                The type of lookback value used for threshold calculation (default: 'average').
                Supported values: 'average', 'median', 'ewm', 'regression'.
            - print_files : bool, optional
                Flag to print file paths during processing.
            - compressed : bool, optional
                Whether the input files are compressed (default: False).
                
        Returns
        -------
        pd.DataFrame
            A DataFrame containing dynamically calculated OHLC bars for the specified symbol.

        Raises
        ------
        ValueError
            If an unsupported method is specified.

        Example
        -------
        symbol = "BTCUSD"
        method = "volume"
        lookback_window = 30
        split_factor = 2
        date_range = ["2023-01-01", "2023-01-10"]
        lookback_type = "average"
        print_files = True
        ohlc_data = self.dynamic_load_transform_ohlc_bars(
            symbol, method, lookback_window, split_factor,
            date_range=date_range, lookback_type=lookback_type, print_files=True
        )
        """
        if method not in ['trades','dollar','volume','volatility','tick_imbalance']:
            raise ValueError(f"Method '{method}' not valid. See documentation for available bar types.")
        
        date_range = kwargs.get("date_range")
        lookback_type = kwargs.get("lookback_type", 'average')
        print_files = kwargs.get("print_files", False)
        compressed = kwargs.get("compressed", False)
        
        symbol_csvs = os.listdir(os.path.join(DATA_DIR, symbol))
        initial_state = None
        # Initialize the result DataFrame
        result_ohlc_bars = pd.DataFrame()
        # Lists to store daily volume values and thresholds.
        self.daily_values = []
        self.thresholds = []
        # Counter var to check against lookback_window.
        self.i = 0
        # Select function to process file chunks.
        self.process_function = self._select_processing_function(method)
        
        for csv in symbol_csvs:
            if date_range:
                if csv.split(".")[0][-10:] not in date_range:
                    continue
            # Set file_path and get the volume ohlc bars.
            file_path = os.path.join(DATA_DIR, symbol, csv)
            self.i += 1
            
            # Select the file opening type based on kwarg.
            if compressed:
                file_context = gzip.open(file_path, 'rt')
            else:
                file_context = open(file_path, 'r')
            
            # Open file and read based on lookback and chunk size.
            with file_context as file:
                # If lookback window not reached, only load the whole file and calculate total volume.
                if self.i <= lookback_window:
                    chunk = pd.read_csv(file)
                    self.daily_values.append(self.daily_value_sum(chunk, method))
                else:
                    # Function to select the daily value used in next bar using the available lookback type options.
                    selected_daily_val = self._calculate_lookback_value(lookback_window, lookback_type)
                    value_threshold = (selected_daily_val / split_factor)
                    self.thresholds.append(value_threshold)
                    
                    if self.chunk_size == 0:
                        chunk = pd.read_csv(file_path)
                        ohlc_bars, final_state = self.process_function(chunk, value_threshold, initial_state)
                        self.daily_values.append(self.daily_value_sum(chunk, method))
                    else:
                        # Read the csv as chunks and process each chunk on its own, concatting into total for file.
                        chunks_volume = 0
                        ohlc_chunk_bars = pd.DataFrame()
                        for chunk in pd.read_csv(file, chunksize=self.chunk_size):
                            # Temporarily add each chunk volume to get the total for day CSV.
                            chunks_volume += self.daily_value_sum(chunk, method)
                            ohlc_bars, final_state = self.process_function(chunk, value_threshold, initial_state)
                            ohlc_chunk_bars = pd.concat([ohlc_chunk_bars, ohlc_bars], axis=0)
                        self.daily_values.append(chunks_volume)
                        ohlc_bars = ohlc_chunk_bars
                    
                    # Concatenate the resulting OHLC bars DataFrame with the overall result
                    result_ohlc_bars = pd.concat([result_ohlc_bars, ohlc_bars], ignore_index=True)
                    # Update the initial state for the next file
                    initial_state = final_state
                    if print_files:
                        print(f"{file_path} -- threshold: {value_threshold:.4f} -- index: {self.i}")
            
        return result_ohlc_bars
