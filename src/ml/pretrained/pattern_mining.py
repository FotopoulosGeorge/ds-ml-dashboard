# src/ml/pretrained/pattern_mining.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# MLxtend for association rules
try:
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

class PatternMiner:
    """
    Pattern mining and association rule discovery
    """
    
    def __init__(self):
        self.frequent_itemsets = None
        self.association_rules = None
        self.transaction_data = None
    
    def render_pattern_mining_tab(self, df):
        """
        Main pattern mining interface
        """
        st.header("🔗 **Pattern Mining & Association Rules**")
        
        if not MLXTEND_AVAILABLE:
            st.error("❌ MLxtend Unavailable")
            st.info("💡 MLxtend provides association rule mining - great for market basket analysis!")
            return
        
        st.markdown("*Discover patterns, associations, and relationships in your data*")
        
        # Pattern mining technique selection
        pattern_type = st.selectbox(
            "**Select Pattern Mining Technique:**",
            [
                "🛒 Market Basket Analysis",
                "🔗 Association Rules Discovery", 
                "📊 Sequential Pattern Mining",
                "🌐 Co-occurrence Analysis",
                "📈 Pattern Visualization"
            ],
            key="pattern_type"
        )
        
        st.markdown("---")
        
        if pattern_type == "🛒 Market Basket Analysis":
            self._market_basket_analysis(df)
        elif pattern_type == "🔗 Association Rules Discovery":
            self._association_rules_discovery(df)
        elif pattern_type == "📊 Sequential Pattern Mining":
            self._sequential_pattern_mining(df)
        elif pattern_type == "🌐 Co-occurrence Analysis":
            self._co_occurrence_analysis(df)
        elif pattern_type == "📈 Pattern Visualization":
            self._pattern_visualization(df)
    
    def _market_basket_analysis(self, df):
        """Market basket analysis for transaction data"""
        st.subheader("🛒 Market Basket Analysis")
        st.markdown("*Find items frequently bought together*")
        
        # Data format detection
        data_format = st.radio(
            "**Data Format:**",
            ["📝 Transaction List", "📊 One-Hot Encoded", "🔄 Auto-Detect"],
            horizontal=True,
            key="basket_format"
        )
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            if data_format == "📝 Transaction List":
                transaction_col = st.selectbox(
                    "**Transaction ID Column:**",
                    df.columns.tolist(),
                    key="transaction_id_col"
                )
                item_col = st.selectbox(
                    "**Item Column:**",
                    df.columns.tolist(),
                    key="item_col"
                )
            else:
                item_columns = st.multiselect(
                    "**Item Columns:**",
                    df.columns.tolist(),
                    key="item_columns"
                )
        
        with config_col2:
            min_support = st.slider(
                "**Minimum Support:**",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Minimum frequency for itemsets",
                key="min_support"
            )
            
            algorithm = st.selectbox(
                "**Algorithm:**",
                ["Apriori", "FP-Growth"],
                help="Apriori: Classic | FP-Growth: Faster for large datasets",
                key="basket_algorithm"
            )
        
        if st.button("🔍 **Find Frequent Itemsets**", type="primary", key="find_itemsets"):
            try:
                with st.spinner('Mining frequent patterns...'):
                    # Prepare transaction data
                    if data_format == "📝 Transaction List":
                        basket_data = self._prepare_transaction_data(df, transaction_col, item_col)
                    else:
                        basket_data = self._prepare_onehot_data(df, item_columns)
                    
                    if basket_data is None or basket_data.empty:
                        st.error("❌ Could not prepare transaction data")
                        return
                    
                    # Mine frequent itemsets
                    if algorithm == "Apriori":
                        frequent_itemsets = apriori(basket_data, min_support=min_support, use_colnames=True)
                    else:  # FP-Growth
                        frequent_itemsets = fpgrowth(basket_data, min_support=min_support, use_colnames=True)
                    
                    if len(frequent_itemsets) == 0:
                        st.warning("⚠️ No frequent itemsets found. Try lowering the minimum support.")
                        return
                    
                    self.frequent_itemsets = frequent_itemsets
                    self.transaction_data = basket_data
                    
                    # Display results
                    self._display_frequent_itemsets(frequent_itemsets)
                    
            except Exception as e:
                st.error(f"❌ Market basket analysis failed: {str(e)}")
                st.info("💡 Check your data format and try adjusting parameters")
    
    def _association_rules_discovery(self, df):
        """Discover association rules from frequent itemsets"""
        st.subheader("🔗 Association Rules Discovery")
        st.markdown("*Find if-then relationships between items*")
        
        if self.frequent_itemsets is None:
            st.info("📝 **First run Market Basket Analysis to find frequent itemsets**")
            return
        
        # Rule configuration
        rule_col1, rule_col2, rule_col3 = st.columns(3)
        
        with rule_col1:
            metric = st.selectbox(
                "**Rule Metric:**",
                ["confidence", "lift", "leverage", "conviction"],
                help="Confidence: P(B|A) | Lift: Confidence/Expected | Leverage: P(AB)-P(A)P(B)",
                key="rule_metric"
            )
        
        with rule_col2:
            min_threshold = st.slider(
                f"**Minimum {metric.title()}:**",
                min_value=0.1 if metric == "lift" else 0.01,
                max_value=10.0 if metric == "lift" else 1.0,
                value=1.2 if metric == "lift" else 0.5,
                step=0.1 if metric == "lift" else 0.01,
                key="min_threshold"
            )
        
        with rule_col3:
            max_len = st.slider(
                "**Max Rule Length:**",
                min_value=2,
                max_value=10,
                value=3,
                key="max_rule_length"
            )
        
        if st.button("🔗 **Generate Association Rules**", type="primary", key="generate_rules"):
            try:
                with st.spinner('Generating association rules...'):
                    # Filter itemsets by length
                    filtered_itemsets = self.frequent_itemsets[
                        self.frequent_itemsets['itemsets'].apply(len) <= max_len
                    ]
                    
                    if len(filtered_itemsets) == 0:
                        st.warning("⚠️ No itemsets found for rule generation")
                        return
                    
                    # Generate rules
                    rules = association_rules(
                        filtered_itemsets, 
                        metric=metric, 
                        min_threshold=min_threshold
                    )
                    
                    if len(rules) == 0:
                        st.warning(f"⚠️ No rules found with {metric} >= {min_threshold}")
                        return
                    
                    self.association_rules = rules
                    
                    # Display rules
                    self._display_association_rules(rules, metric)
                    
            except Exception as e:
                st.error(f"❌ Rule generation failed: {str(e)}")
    
    def _sequential_pattern_mining(self, df):
        """Sequential pattern mining for time-ordered data"""
        st.subheader("📊 Sequential Pattern Mining")
        st.markdown("*Find patterns in time-ordered sequences*")
        
        # Data configuration
        seq_col1, seq_col2, seq_col3 = st.columns(3)
        
        with seq_col1:
            sequence_id_col = st.selectbox(
                "**Sequence ID:**",
                df.columns.tolist(),
                help="Column identifying different sequences (e.g., customer_id)",
                key="seq_id_col"
            )
        
        with seq_col2:
            timestamp_col = st.selectbox(
                "**Timestamp Column:**",
                df.select_dtypes(include=['datetime64', 'object']).columns.tolist(),
                key="timestamp_col"
            )
        
        with seq_col3:
            event_col = st.selectbox(
                "**Event/Item Column:**",
                df.columns.tolist(),
                key="event_col"
            )
        
        # Parameters
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            min_seq_length = st.slider("Min Sequence Length:", 2, 10, 3, key="min_seq_length")
            max_seq_length = st.slider("Max Sequence Length:", 3, 20, 5, key="max_seq_length")
        
        with param_col2:
            min_frequency = st.slider("Min Pattern Frequency:", 1, 100, 5, key="min_frequency")
            window_size = st.slider("Time Window (hours):", 1, 168, 24, key="window_size")
        
        if st.button("🔍 **Mine Sequential Patterns**", type="primary", key="mine_sequences"):
            try:
                with st.spinner('Mining sequential patterns...'):
                    sequences = self._extract_sequences(
                        df, sequence_id_col, timestamp_col, event_col, window_size
                    )
                    
                    patterns = self._find_sequential_patterns(
                        sequences, min_seq_length, max_seq_length, min_frequency
                    )
                    
                    self._display_sequential_patterns(patterns)
                    
            except Exception as e:
                st.error(f"❌ Sequential pattern mining failed: {str(e)}")
    
    def _co_occurrence_analysis(self, df):
        """Co-occurrence analysis for finding item relationships"""
        st.subheader("🌐 Co-occurrence Analysis")
        st.markdown("*Analyze how often items appear together*")
        
        # Column selection
        cooc_col1, cooc_col2 = st.columns(2)
        
        with cooc_col1:
            primary_col = st.selectbox(
                "**Primary Items:**",
                df.columns.tolist(),
                key="primary_cooc_col"
            )
        
        with cooc_col2:
            secondary_col = st.selectbox(
                "**Secondary Items:**",
                df.columns.tolist(),
                key="secondary_cooc_col"
            )
        
        # Analysis parameters
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            min_cooccurrence = st.slider(
                "**Minimum Co-occurrence:**",
                min_value=1,
                max_value=50,
                value=3,
                key="min_cooccurrence"
            )
        
        with analysis_col2:
            top_n_pairs = st.slider(
                "**Top N Pairs to Show:**",
                min_value=5,
                max_value=100,
                value=20,
                key="top_n_pairs"
            )
        
        if st.button("🌐 **Analyze Co-occurrence**", type="primary", key="analyze_cooccurrence"):
            try:
                with st.spinner('Analyzing co-occurrences...'):
                    cooccurrence_matrix = self._calculate_cooccurrence(
                        df, primary_col, secondary_col, min_cooccurrence
                    )
                    
                    if cooccurrence_matrix is None:
                        st.warning("⚠️ No significant co-occurrences found")
                        return
                    
                    self._display_cooccurrence_results(cooccurrence_matrix, top_n_pairs)
                    
            except Exception as e:
                st.error(f"❌ Co-occurrence analysis failed: {str(e)}")
    
    def _pattern_visualization(self, df):
        """Visualize discovered patterns and rules"""
        st.subheader("📈 Pattern Visualization")
        
        if self.association_rules is None and self.frequent_itemsets is None:
            st.info("📝 **First discover patterns using other tabs to enable visualization**")
            return
        
        viz_type = st.selectbox(
            "**Visualization Type:**",
            ["📊 Rule Metrics Scatter", "🌐 Network Graph", "📈 Support vs Confidence", "🔥 Heatmap"],
            key="viz_type"
        )
        
        if viz_type == "📊 Rule Metrics Scatter" and self.association_rules is not None:
            self._plot_rule_metrics_scatter()
        elif viz_type == "🌐 Network Graph" and self.association_rules is not None:
            self._plot_network_graph()
        elif viz_type == "📈 Support vs Confidence" and self.association_rules is not None:
            self._plot_support_confidence()
        elif viz_type == "🔥 Heatmap" and self.frequent_itemsets is not None:
            self._plot_pattern_heatmap()
        else:
            st.warning("⚠️ Selected visualization requires association rules or frequent itemsets")
    
    # Helper methods
    def _prepare_transaction_data(self, df, transaction_col, item_col):
        """Prepare transaction data for basket analysis"""
        # Group by transaction and create lists of items
        transactions = df.groupby(transaction_col)[item_col].apply(list).tolist()
        
        # Use TransactionEncoder to convert to one-hot encoded format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        return basket_df
    
    def _prepare_onehot_data(self, df, item_columns):
        """Prepare one-hot encoded data"""
        if not item_columns:
            return None
        
        basket_df = df[item_columns].copy()
        
        # Convert to boolean if not already
        for col in basket_df.columns:
            if basket_df[col].dtype != bool:
                basket_df[col] = basket_df[col].astype(bool)
        
        return basket_df
    
    def _display_frequent_itemsets(self, frequent_itemsets):
        """Display frequent itemsets results"""
        st.success(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
        
        # Summary metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Itemsets", len(frequent_itemsets))
        with metric_col2:
            avg_support = frequent_itemsets['support'].mean()
            st.metric("Avg Support", f"{avg_support:.3f}")
        with metric_col3:
            max_length = frequent_itemsets['itemsets'].apply(len).max()
            st.metric("Max Length", max_length)
        with metric_col4:
            single_items = len(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1])
            st.metric("Single Items", single_items)
        
        # Display itemsets table
        st.subheader("📋 Frequent Itemsets")
        
        # Prepare display data
        display_itemsets = frequent_itemsets.copy()
        display_itemsets['items'] = display_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
        display_itemsets['length'] = display_itemsets['itemsets'].apply(len)
        display_itemsets = display_itemsets[['items', 'length', 'support']].sort_values('support', ascending=False)
        
        st.dataframe(display_itemsets, use_container_width=True)
        
        # Support distribution
        st.subheader("📊 Support Distribution")
        fig = px.histogram(
            frequent_itemsets, 
            x='support', 
            nbins=20,
            title='Distribution of Support Values'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        csv_data = display_itemsets.to_csv(index=False)
        st.download_button(
            label="📥 **Download Frequent Itemsets**",
            data=csv_data,
            file_name=f"frequent_itemsets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    def _display_association_rules(self, rules, metric):
        """Display association rules results"""
        st.success(f"✅ Generated {len(rules)} association rules")
        
        # Summary metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Rules", len(rules))
        with metric_col2:
            avg_confidence = rules['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        with metric_col3:
            avg_lift = rules['lift'].mean()
            st.metric("Avg Lift", f"{avg_lift:.3f}")
        with metric_col4:
            strong_rules = len(rules[rules['lift'] > 1])
            st.metric("Strong Rules (Lift>1)", strong_rules)
        
        # Display rules table
        st.subheader("🔗 Association Rules")
        
        # Prepare display data
        display_rules = rules.copy()
        display_rules['antecedents_str'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents_str'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Select columns for display
        display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
        display_rules_clean = display_rules[display_cols].copy()
        display_rules_clean.columns = ['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift']
        display_rules_clean = display_rules_clean.sort_values('Lift', ascending=False)
        
        st.dataframe(display_rules_clean, use_container_width=True)
        
        # Top rules by metric
        st.subheader(f"🏆 Top Rules by {metric.title()}")
        top_rules = display_rules_clean.head(10)
        
        for idx, rule in top_rules.iterrows():
            with st.expander(f"📜 {rule['Antecedents']} → {rule['Consequents']}"):
                rule_col1, rule_col2, rule_col3 = st.columns(3)
                
                with rule_col1:
                    st.metric("Support", f"{rule['Support']:.3f}")
                with rule_col2:
                    st.metric("Confidence", f"{rule['Confidence']:.3f}")
                with rule_col3:
                    st.metric("Lift", f"{rule['Lift']:.3f}")
                
                # Rule interpretation
                if rule['Lift'] > 1:
                    st.success("✅ **Strong positive association** - Items appear together more than expected")
                elif rule['Lift'] < 1:
                    st.warning("⚠️ **Negative association** - Items appear together less than expected")
                else:
                    st.info("ℹ️ **No association** - Items appear together as expected by chance")
        
        # Download rules
        csv_data = display_rules_clean.to_csv(index=False)
        st.download_button(
            label="📥 **Download Association Rules**",
            data=csv_data,
            file_name=f"association_rules_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    def _extract_sequences(self, df, seq_id_col, timestamp_col, event_col, window_size):
        """Extract sequences from time-ordered data"""
        # Convert timestamp to datetime if needed
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[timestamp_col]):
            df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
        
        # Sort by sequence ID and timestamp
        df_copy = df_copy.sort_values([seq_id_col, timestamp_col])
        
        sequences = []
        for seq_id, group in df_copy.groupby(seq_id_col):
            # Group events within time windows
            events = group[event_col].tolist()
            timestamps = group[timestamp_col].tolist()
            
            # Simple sequence extraction (can be enhanced)
            if len(events) >= 2:
                sequences.append(events)
        
        return sequences
    
    def _find_sequential_patterns(self, sequences, min_length, max_length, min_frequency):
        """Find frequent sequential patterns"""
        # Simple implementation - can be enhanced with more sophisticated algorithms
        pattern_counts = {}
        
        for sequence in sequences:
            # Generate all subsequences
            for length in range(min_length, min(max_length + 1, len(sequence) + 1)):
                for i in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[i:i+length])
                    pattern_counts[subseq] = pattern_counts.get(subseq, 0) + 1
        
        # Filter by minimum frequency
        frequent_patterns = {
            pattern: count for pattern, count in pattern_counts.items() 
            if count >= min_frequency
        }
        
        return frequent_patterns
    
    def _display_sequential_patterns(self, patterns):
        """Display sequential patterns results"""
        if not patterns:
            st.warning("⚠️ No sequential patterns found with current parameters")
            return
        
        st.success(f"✅ Found {len(patterns)} sequential patterns")
        
        # Convert to dataframe for display
        pattern_data = []
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            pattern_data.append({
                'Pattern': ' → '.join(pattern),
                'Length': len(pattern),
                'Frequency': count
            })
        
        pattern_df = pd.DataFrame(pattern_data)
        st.dataframe(pattern_df, use_container_width=True)
        
        # Pattern length distribution
        length_dist = pattern_df['Length'].value_counts().sort_index()
        fig = px.bar(
            x=length_dist.index,
            y=length_dist.values,
            title='Sequential Pattern Length Distribution',
            labels={'x': 'Pattern Length', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _calculate_cooccurrence(self, df, primary_col, secondary_col, min_cooccurrence):
        """Calculate co-occurrence matrix"""
        # Create co-occurrence pairs
        cooccurrence_pairs = []
        
        if primary_col == secondary_col:
            # Self co-occurrence within the same column
            for val1 in df[primary_col].unique():
                for val2 in df[primary_col].unique():
                    if val1 != val2:
                        # Count how often they appear in the same context (simple implementation)
                        count = len(df[(df[primary_col] == val1) & (df[primary_col] == val2)])
                        if count >= min_cooccurrence:
                            cooccurrence_pairs.append({
                                'Item1': val1,
                                'Item2': val2,
                                'Cooccurrence': count
                            })
        else:
            # Cross-column co-occurrence
            cooc_counts = df.groupby([primary_col, secondary_col]).size().reset_index(name='Cooccurrence')
            cooc_counts = cooc_counts[cooc_counts['Cooccurrence'] >= min_cooccurrence]
            
            for _, row in cooc_counts.iterrows():
                cooccurrence_pairs.append({
                    'Item1': row[primary_col],
                    'Item2': row[secondary_col],
                    'Cooccurrence': row['Cooccurrence']
                })
        
        if not cooccurrence_pairs:
            return None
        
        return pd.DataFrame(cooccurrence_pairs)
    
    def _display_cooccurrence_results(self, cooccurrence_matrix, top_n):
        """Display co-occurrence analysis results"""
        st.success(f"✅ Found {len(cooccurrence_matrix)} co-occurrence pairs")
        
        # Top pairs
        top_pairs = cooccurrence_matrix.nlargest(top_n, 'Cooccurrence')
        st.dataframe(top_pairs, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            top_pairs.head(20),
            x='Cooccurrence',
            y=top_pairs.head(20)['Item1'] + ' + ' + top_pairs.head(20)['Item2'],
            orientation='h',
            title='Top Co-occurrence Pairs'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_rule_metrics_scatter(self):
        """Plot rule metrics scatter plot"""
        if self.association_rules is None:
            return
        
        fig = px.scatter(
            self.association_rules,
            x='support',
            y='confidence',
            size='lift',
            title='Association Rules: Support vs Confidence (bubble size = lift)',
            labels={'support': 'Support', 'confidence': 'Confidence'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_support_confidence(self):
        """Plot support vs confidence"""
        if self.association_rules is None:
            return
        
        fig = px.scatter(
            self.association_rules,
            x='support',
            y='confidence',
            color='lift',
            title='Support vs Confidence colored by Lift',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_network_graph(self):
        """Plot network graph of associations"""
        if self.association_rules is None:
            return
        
        st.info("🌐 **Network visualization** - Shows item relationships")
        
        # Create a simple network visualization using plotly
        # This is a simplified version - can be enhanced with networkx
        
        # Get top rules by lift
        top_rules = self.association_rules.nlargest(20, 'lift')
        
        # Create nodes and edges
        nodes = set()
        edges = []
        
        for _, rule in top_rules.iterrows():
            antecedent = ', '.join(list(rule['antecedents']))
            consequent = ', '.join(list(rule['consequents']))
            
            nodes.add(antecedent)
            nodes.add(consequent)
            
            edges.append({
                'source': antecedent,
                'target': consequent,
                'weight': rule['lift']
            })
        
        # Simple network info display
        st.write(f"**Network contains {len(nodes)} nodes and {len(edges)} edges**")
        
        # Display as table (simplified)
        network_df = pd.DataFrame(edges)
        if not network_df.empty:
            st.dataframe(network_df, use_container_width=True)
    
    def _plot_pattern_heatmap(self):
        """Plot pattern heatmap"""
        if self.frequent_itemsets is None:
            return
        
        # Create a simple heatmap of item frequencies
        st.info("🔥 **Pattern Heatmap** - Shows item frequency patterns")
        
        # This is a simplified implementation
        st.write("Heatmap visualization would show item co-occurrence patterns")
        st.caption("Advanced heatmap visualization can be implemented with more complex data structures")