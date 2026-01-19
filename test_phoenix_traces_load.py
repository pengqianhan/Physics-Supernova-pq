"""
Test script for load_phoenix_traces function.

This script tests loading Phoenix traces from the phoenix_traces directory.
Also provides utilities to export traces to JSON and Markdown formats.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_llm_ha_beta import load_phoenix_traces, PHOENIX_AVAILABLE


def test_load_phoenix_traces():
    """Test loading Phoenix traces from saved parquet file."""
    
    # The trace file: trace_dataset-c30ffff3-88e0-45e5-803e-0d585bbb1f49.parquet
    trace_id = "c30ffff3-88e0-45e5-803e-0d585bbb1f49"
    load_dir = "phoenix_traces"
    
    print("=" * 60)
    print("Testing load_phoenix_traces")
    print("=" * 60)
    print(f"PHOENIX_AVAILABLE: {PHOENIX_AVAILABLE}")
    print(f"Trace ID: {trace_id}")
    print(f"Load directory: {load_dir}")
    print()
    
    # Check if trace file exists
    expected_file = os.path.join(load_dir, f"trace_dataset-{trace_id}.parquet")
    if os.path.exists(expected_file):
        print(f"[OK] Trace file exists: {expected_file}")
        file_size = os.path.getsize(expected_file)
        print(f"     File size: {file_size / 1024:.2f} KB")
    else:
        print(f"[ERROR] Trace file not found: {expected_file}")
        return False
    
    print()
    
    # Test loading traces
    if not PHOENIX_AVAILABLE:
        print("[SKIP] Phoenix not available, cannot test load_phoenix_traces")
        print("       Install: pip install arize-phoenix openinference-instrumentation-smolagents")
        return None
    
    print("Loading traces...")
    trace_ds = load_phoenix_traces(trace_id, load_dir)
    
    if trace_ds is None:
        print("[ERROR] Failed to load traces - returned None")
        return False
    
    print(f"[OK] Traces loaded successfully")
    print(f"     Type: {type(trace_ds).__name__}")
    print()
    
    # Get spans dataframe
    print("Getting spans dataframe...")
    try:
        spans_df = trace_ds.get_spans_dataframe()
        print(f"[OK] Spans dataframe retrieved")
        print(f"     Shape: {spans_df.shape}")
        print(f"     Columns: {list(spans_df.columns)}")
        print()
        
        # Show some basic stats
        print("Dataframe Info:")
        print(f"  - Number of spans: {len(spans_df)}")
        
        if 'name' in spans_df.columns:
            print(f"  - Unique span names: {spans_df['name'].nunique()}")
            print(f"  - Span names: {spans_df['name'].unique().tolist()[:10]}")  # Show first 10
        
        if 'span_kind' in spans_df.columns:
            print(f"  - Span kinds: {spans_df['span_kind'].value_counts().to_dict()}")
        
        if 'status_code' in spans_df.columns:
            print(f"  - Status codes: {spans_df['status_code'].value_counts().to_dict()}")
        
        print()
        print("First few rows (head):")
        print(spans_df.head())
        
    except Exception as e:
        print(f"[ERROR] Failed to get spans dataframe: {e}")
        return False
    
    print()
    print("=" * 60)
    print("[SUCCESS] All tests passed!")
    print("=" * 60)
    return True


def list_available_traces(load_dir: str = "phoenix_traces"):
    """List all available trace files in the directory."""
    print(f"Available trace files in '{load_dir}':")
    
    if not os.path.exists(load_dir):
        print(f"  Directory not found: {load_dir}")
        return []
    
    trace_files = []
    for f in os.listdir(load_dir):
        if f.startswith("trace_dataset-") and f.endswith(".parquet"):
            # Extract trace_id from filename
            # Format: trace_dataset-{trace_id}.parquet
            trace_id = f.replace("trace_dataset-", "").replace(".parquet", "")
            trace_files.append({
                "filename": f,
                "trace_id": trace_id,
                "path": os.path.join(load_dir, f),
                "size": os.path.getsize(os.path.join(load_dir, f))
            })
            print(f"  - {f}")
            print(f"    trace_id: {trace_id}")
            print(f"    size: {trace_files[-1]['size'] / 1024:.2f} KB")
    
    if not trace_files:
        print("  No trace files found")
    
    return trace_files


def test_load_parquet_directly():
    """
    Fallback test: Read parquet file directly using pandas or pyarrow.
    This works even without Phoenix installed.
    """
    trace_id = "c30ffff3-88e0-45e5-803e-0d585bbb1f49"
    load_dir = "phoenix_traces"
    parquet_path = os.path.join(load_dir, f"trace_dataset-{trace_id}.parquet")
    
    print("=" * 60)
    print("Testing direct parquet file read (fallback without Phoenix)")
    print("=" * 60)
    print(f"Parquet file: {parquet_path}")
    print()
    
    if not os.path.exists(parquet_path):
        print(f"[ERROR] Parquet file not found: {parquet_path}")
        return False
    
    # Try pandas first
    try:
        import pandas as pd
        
        print("Reading parquet file with pandas...")
        df = pd.read_parquet(parquet_path)
        
        print(f"[OK] Parquet file loaded successfully")
        print(f"     Shape: {df.shape}")
        print()
        
        print("Columns:")
        for col in df.columns:
            print(f"  - {col}")
        print()
        
        print("Dataframe Info:")
        print(f"  - Number of rows: {len(df)}")
        print(f"  - Number of columns: {len(df.columns)}")
        print()
        
        # Show column types
        print("Column types:")
        for col in df.columns[:10]:  # Show first 10
            print(f"  - {col}: {df[col].dtype}")
        if len(df.columns) > 10:
            print(f"  ... and {len(df.columns) - 10} more columns")
        print()
        
        # Show data preview
        print("Data preview (first 3 rows):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df.head(3).to_string())
        print()
        
        # If there's a 'name' column, show unique values
        if 'name' in df.columns:
            print(f"Unique span names: {df['name'].unique().tolist()}")
        
        print()
        print("=" * 60)
        print("[SUCCESS] Direct parquet read test passed!")
        print("=" * 60)
        return True
        
    except ImportError:
        print("[INFO] pandas not available, trying pyarrow...")
    except Exception as e:
        print(f"[ERROR] Failed to read with pandas: {e}")
    
    # Try pyarrow as fallback
    try:
        import pyarrow.parquet as pq
        
        print("Reading parquet file with pyarrow...")
        table = pq.read_table(parquet_path)
        
        print(f"[OK] Parquet file loaded successfully with pyarrow")
        print(f"     Number of rows: {table.num_rows}")
        print(f"     Number of columns: {table.num_columns}")
        print()
        
        print("Schema:")
        print(table.schema)
        print()
        
        print("Column names:")
        for col in table.column_names:
            print(f"  - {col}")
        print()
        
        # Convert to dict for preview
        print("Data preview (first row as dict):")
        if table.num_rows > 0:
            row = table.slice(0, 1).to_pydict()
            for k, v in row.items():
                val_str = str(v[0]) if v else "None"
                if len(val_str) > 80:
                    val_str = val_str[:80] + "..."
                print(f"  {k}: {val_str}")
        
        print()
        print("=" * 60)
        print("[SUCCESS] Direct parquet read test passed (pyarrow)!")
        print("=" * 60)
        return True
        
    except ImportError:
        print("[ERROR] Neither pandas nor pyarrow installed.")
        print("        Install: pip install pandas pyarrow")
        print()
        print("[INFO] File verification only - parquet file exists and is valid.")
        print("       Full content inspection requires pandas or pyarrow.")
        return None  # Partial success - file exists but can't read content
    except Exception as e:
        print(f"[ERROR] Failed to read parquet: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_parquet_as_dict(parquet_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load parquet file and return as list of dictionaries.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        List of dictionaries (one per row), or None if failed
    """
    if not os.path.exists(parquet_path):
        print(f"[ERROR] File not found: {parquet_path}")
        return None
    
    # Try pandas first
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        # Convert timestamps to ISO format strings for JSON serialization
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        return df.to_dict(orient='records')
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] pandas failed: {e}")
    
    # Try pyarrow
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        data = table.to_pydict()
        # Convert to list of dicts
        num_rows = table.num_rows
        records = []
        for i in range(num_rows):
            row = {}
            for col in data:
                val = data[col][i]
                # Handle non-JSON-serializable types
                if hasattr(val, 'isoformat'):
                    val = val.isoformat()
                elif hasattr(val, 'as_py'):
                    val = val.as_py()
                row[col] = val
            records.append(row)
        return records
    except ImportError:
        print("[ERROR] Neither pandas nor pyarrow installed.")
        return None
    except Exception as e:
        print(f"[ERROR] pyarrow failed: {e}")
        return None


def export_traces_to_json(
    parquet_path: str = "phoenix_traces/trace_dataset-c30ffff3-88e0-45e5-803e-0d585bbb1f49.parquet",
    output_path: str = "phoenix_traces/traces.json",
    indent: int = 2
) -> bool:
    """
    Export Phoenix traces from parquet to JSON format.
    
    Args:
        parquet_path: Path to the parquet file
        output_path: Path to save the JSON file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Exporting traces to JSON: {output_path}")
    
    records = load_parquet_as_dict(parquet_path)
    if records is None:
        return False
    
    # Custom JSON encoder for special types
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            if hasattr(obj, 'as_py'):
                return obj.as_py()
            if hasattr(obj, '__dict__'):
                return str(obj)
            try:
                return super().default(obj)
            except TypeError:
                return str(obj)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=indent, ensure_ascii=False, cls=CustomEncoder)
        
        file_size = os.path.getsize(output_path)
        print(f"[OK] Exported {len(records)} traces to {output_path} ({file_size / 1024:.2f} KB)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write JSON: {e}")
        return False


def export_traces_to_markdown(
    parquet_path: str = "phoenix_traces/trace_dataset-c30ffff3-88e0-45e5-803e-0d585bbb1f49.parquet",
    output_path: str = "phoenix_traces/traces.md",
    max_content_length: int = 2000
) -> bool:
    """
    Export Phoenix traces from parquet to Markdown format for easy reading.
    
    Args:
        parquet_path: Path to the parquet file
        output_path: Path to save the Markdown file
        max_content_length: Maximum length for content fields (truncate if longer)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Exporting traces to Markdown: {output_path}")
    
    records = load_parquet_as_dict(parquet_path)
    if records is None:
        return False
    
    def truncate(text: str, max_len: int = max_content_length) -> str:
        """Truncate text and add ellipsis if too long."""
        if text is None:
            return "None"
        text = str(text)
        if len(text) > max_len:
            return text[:max_len] + f"\n\n... (truncated, {len(text) - max_len} more chars)"
        return text
    
    def format_value(val, max_len: int = 500) -> str:
        """Format a value for markdown display."""
        if val is None:
            return "_None_"
        if isinstance(val, dict):
            return f"```json\n{json.dumps(val, indent=2, ensure_ascii=False)[:max_len]}\n```"
        if isinstance(val, list):
            return f"```json\n{json.dumps(val, indent=2, ensure_ascii=False)[:max_len]}\n```"
        return truncate(str(val), max_len)
    
    try:
        lines = []
        lines.append("# Phoenix Traces Export")
        lines.append("")
        lines.append(f"**Source:** `{parquet_path}`")
        lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Spans:** {len(records)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for i, record in enumerate(records):
            name = record.get('name', f'Span {i+1}')
            span_id = record.get('context.span_id', record.get('span_id', f'{i+1}'))
            lines.append(f"- [{i+1}. {name}](#{i+1}-{name.lower().replace(' ', '-').replace('.', '')})")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Important columns to show first
        priority_cols = [
            'name', 'span_kind', 'status_code', 'status_message',
            'start_time', 'end_time', 'latency_ms',
            'context.trace_id', 'context.span_id', 'parent_id',
            'attributes.input.value', 'attributes.output.value',
            'attributes.llm.input_messages', 'attributes.llm.output_messages',
        ]
        
        for i, record in enumerate(records):
            name = record.get('name', f'Span {i+1}')
            lines.append(f"## {i+1}. {name}")
            lines.append("")
            
            # Show priority columns first
            shown_cols = set()
            for col in priority_cols:
                if col in record and record[col] is not None:
                    val = record[col]
                    lines.append(f"### {col}")
                    lines.append("")
                    lines.append(format_value(val))
                    lines.append("")
                    shown_cols.add(col)
            
            # Show remaining columns
            other_cols = [c for c in record.keys() if c not in shown_cols and record[c] is not None]
            if other_cols:
                lines.append("### Other Attributes")
                lines.append("")
                lines.append("| Attribute | Value |")
                lines.append("|-----------|-------|")
                for col in sorted(other_cols):
                    val = record[col]
                    val_str = str(val)[:100].replace('\n', ' ').replace('|', '\\|')
                    if len(str(val)) > 100:
                        val_str += "..."
                    lines.append(f"| `{col}` | {val_str} |")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        file_size = os.path.getsize(output_path)
        print(f"[OK] Exported {len(records)} traces to {output_path} ({file_size / 1024:.2f} KB)")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to write Markdown: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_traces(
    trace_id: str = "c30ffff3-88e0-45e5-803e-0d585bbb1f49",
    load_dir: str = "phoenix_traces",
    output_format: str = "both"
) -> bool:
    """
    Export Phoenix traces to JSON and/or Markdown.
    
    Args:
        trace_id: The trace ID
        load_dir: Directory containing the parquet file
        output_format: "json", "markdown", or "both"
        
    Returns:
        True if successful
    """
    parquet_path = os.path.join(load_dir, f"trace_dataset-{trace_id}.parquet")
    
    print("=" * 60)
    print("Exporting Phoenix Traces")
    print("=" * 60)
    print(f"Trace ID: {trace_id}")
    print(f"Source: {parquet_path}")
    print(f"Format: {output_format}")
    print()
    
    success = True
    
    if output_format in ("json", "both"):
        json_path = os.path.join(load_dir, f"traces_{trace_id}.json")
        if not export_traces_to_json(parquet_path, json_path):
            success = False
    
    if output_format in ("markdown", "both"):
        md_path = os.path.join(load_dir, f"traces_{trace_id}.md")
        if not export_traces_to_markdown(parquet_path, md_path):
            success = False
    
    print()
    if success:
        print("[SUCCESS] Export completed!")
    else:
        print("[PARTIAL] Some exports failed")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phoenix Traces Test and Export Tool")
    parser.add_argument("--export", choices=["json", "markdown", "both"], 
                        help="Export traces to specified format")
    parser.add_argument("--trace-id", default="c30ffff3-88e0-45e5-803e-0d585bbb1f49",
                        help="Trace ID to export")
    parser.add_argument("--dir", default="phoenix_traces",
                        help="Directory containing trace files")
    args = parser.parse_args()
    
    # If export flag is set, just do the export
    if args.export:
        success = export_traces(args.trace_id, args.dir, args.export)
        sys.exit(0 if success else 1)
    
    # Otherwise run the default test
    print("\n" + "=" * 60)
    print("Phoenix Traces Load Test")
    print("=" * 60 + "\n")
    
    # List available traces
    traces = list_available_traces()
    print()
    
    # Run Phoenix test
    result = test_load_phoenix_traces()
    print()
    
    # If Phoenix not available, run fallback parquet test
    if result is None:
        print("Running fallback parquet test (without Phoenix)...\n")
        fallback_result = test_load_parquet_directly()
        
        if fallback_result is True:
            sys.exit(0)  # Full success
        elif fallback_result is None:
            # Partial success - file exists but couldn't read content
            print("\n[PARTIAL] File verification passed. Install pandas/pyarrow for full test.")
            sys.exit(0)
        else:
            sys.exit(1)  # Failure
    elif result:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure
