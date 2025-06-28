import streamlit as st
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
import av
import cv2
from PIL import Image
import threading
import time
import os
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
CSV_PATH = "/Users/path/to/nutrition_table.csv"
MODEL_PATH = "/Users/path/to/model.pt"
CONFIDENCE_THRESHOLD = 0.2
FRAME_CAPTURE_INTERVAL = 2  # seconds

# Create a file-based communication method for detection results
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
DETECTION_FILE = os.path.join(TEMP_DIR, "latest_detection.txt")
FRAME_FILE = os.path.join(TEMP_DIR, "latest_frame.jpg")
LOG_FILE = os.path.join(TEMP_DIR, "detection_log.json")
CALORIE_LOG_FILE = os.path.join(TEMP_DIR, "calorie_intake_log.json")  # New calorie log file

# --- Session state initialization ---
if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = time.time() - FRAME_CAPTURE_INTERVAL
if 'frame_counter' not in st.session_state:
    st.session_state.frame_counter = 0
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0
if 'show_logs' not in st.session_state:
    st.session_state.show_logs = False

# --- Caching & Model Loading ---
@st.cache_resource
def load_model():
    """Loads the YOLO model from the specified path."""
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = YOLO(MODEL_PATH).to(device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_nutrition_data():
    """Loads and preprocesses the nutrition data from a CSV file."""
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.lower().strip() for c in df.columns]
        df['food'] = df['food'].str.lower().str.strip()
        return df
    except FileNotFoundError:
        st.error(f"Nutrition data file not found at: {CSV_PATH}")
        return pd.DataFrame()

# --- Helper Functions ---
def get_nutrition_info(food, nutrition_df):
    """Retrieves nutrition information for a given food item."""
    row = nutrition_df[nutrition_df['food'].str.lower() == food.lower()]
    if not row.empty:
        row = row.iloc[0]
        return {
            "Calories": row.get('calories', 0),
            "Protein": row.get('protein', 'N/A'),
            "Fat": row.get('fat', 'N/A'),
            "Sodium": row.get('sodium', 'N/A'),
            "Fiber": row.get('fiber', 'N/A'),
            "Carbohydrates": row.get('carbo', 'N/A'),
            "Sugars": row.get('sugars', 'N/A'),
            "Potassium": row.get('potassium', 'N/A'),
            "Vitamin C": row.get('vitamin_c', 'N/A'),
            "Iron": row.get('iron', 'N/A'),
            "Calcium": row.get('calcium', 'N/A'),
            "Ingredients": row.get('ingredients', 'N/A'),
            "Recipe": row.get('recipe', '#')
        }
    return None

def calculate_total_calories(detections, nutrition_df):
    """Calculate total calories from detected foods"""
    total_calories = 0
    food_calories = {}
    
    for food, count in detections.items():
        nutrition_info = get_nutrition_info(food, nutrition_df)
        if nutrition_info and nutrition_info['Calories'] != 'N/A':
            try:
                calories_per_item = float(nutrition_info['Calories'])
                food_total_calories = calories_per_item * count
                food_calories[food] = {
                    'count': count,
                    'calories_per_item': calories_per_item,
                    'total_calories': food_total_calories
                }
                total_calories += food_total_calories
            except (ValueError, TypeError):
                food_calories[food] = {
                    'count': count,
                    'calories_per_item': 0,
                    'total_calories': 0
                }
    
    return total_calories, food_calories

def process_image_and_detect(frame, model, device):
    """Runs YOLO prediction on a single frame and returns results."""
    results = model.predict(source=frame, save=False, device=device, verbose=False)
    detections = results[0]
    detected_classes = {}
    for box in detections.boxes:
        conf = float(box.conf[0])
        if conf >= CONFIDENCE_THRESHOLD:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_classes[label] = detected_classes.get(label, 0) + 1
    annotated_frame = detections.plot()
    return annotated_frame, detected_classes

def write_detections_to_file(detections):
    """Write detection results to a temporary file for inter-process communication"""
    try:
        with open(DETECTION_FILE, 'w') as f:
            for food, count in detections.items():
                f.write(f"{food}:{count}\n")
                
        # Also log detections with timestamp for history
        log_detection(detections)
    except Exception as e:
        st.error(f"Error writing detections: {e}")

def log_detection(detections):
    """Log detections with timestamp and calorie information to history file"""
    if not detections:
        return
        
    nutrition_df = load_nutrition_data()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_calories, food_calories = calculate_total_calories(detections, nutrition_df)
    
    log_entry = {
        "timestamp": timestamp,
        "detections": detections,
        "total_calories": total_calories,
        "food_calories": food_calories
    }
    
    # Load existing logs
    log_data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            log_data = []
    
    # Add new entry and save
    log_data.append(log_entry)
    
    # Keep only the last 100 entries to avoid file getting too large
    log_data = log_data[-100:]
    
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f)
    
    # Also log to calorie intake history
    log_calorie_intake(timestamp, total_calories, food_calories)

def log_calorie_intake(timestamp, total_calories, food_calories):
    """Log calorie intake to a separate calorie history file"""
    date = timestamp.split()[0]  # Extract date part
    
    # Load existing calorie logs
    calorie_data = {}
    if os.path.exists(CALORIE_LOG_FILE):
        try:
            with open(CALORIE_LOG_FILE, 'r') as f:
                calorie_data = json.load(f)
        except json.JSONDecodeError:
            calorie_data = {}
    
    # Initialize date if not exists
    if date not in calorie_data:
        calorie_data[date] = {
            "total_calories": 0,
            "meals": [],
            "food_breakdown": {}
        }
    
    # Add this meal to the date
    meal_entry = {
        "time": timestamp.split()[1],  # Extract time part
        "calories": total_calories,
        "foods": food_calories
    }
    
    calorie_data[date]["meals"].append(meal_entry)
    calorie_data[date]["total_calories"] += total_calories
    
    # Update food breakdown for the day
    for food, info in food_calories.items():
        if food not in calorie_data[date]["food_breakdown"]:
            calorie_data[date]["food_breakdown"][food] = 0
        calorie_data[date]["food_breakdown"][food] += info['total_calories']
    
    # Save updated calorie data
    with open(CALORIE_LOG_FILE, 'w') as f:
        json.dump(calorie_data, f)

def get_detection_logs():
    """Read detection logs from file"""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def get_calorie_logs():
    """Read calorie intake logs from file"""
    if os.path.exists(CALORIE_LOG_FILE):
        try:
            with open(CALORIE_LOG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def read_detections_from_file():
    """Read detection results from the temporary file"""
    detections = {}
    try:
        if os.path.exists(DETECTION_FILE):
            with open(DETECTION_FILE, 'r') as f:
                for line in f:
                    if ':' in line:
                        food, count = line.strip().split(':', 1)
                        detections[food] = int(count)
    except Exception as e:
        st.error(f"Error reading detections: {e}")
    return detections

def save_frame_to_file(frame):
    """Save frame to file for displaying"""
    try:
        cv2.imwrite(FRAME_FILE, frame)
    except Exception as e:
        st.error(f"Error saving frame: {e}")

def load_frame_from_file():
    """Load frame from file"""
    if os.path.exists(FRAME_FILE):
        return cv2.imread(FRAME_FILE)
    return None

# --- Main App ---
st.set_page_config(page_title="Food Detection & Calorie Tracker", layout="wide")

# Application with tabs for better UI organization
tab1, tab2, tab3 = st.tabs(["🍲 Detection", "📋 Detection History", "📊 Calorie Tracker"])

with tab1:
    st.title("🍲 Food Detection & Nutrition Info")

    # Load model and data
    model, device = load_model()
    nutrition_df = load_nutrition_data()

    # Exit if model or data failed to load
    if model is None or nutrition_df.empty:
        st.warning("Application cannot start due to loading errors. Please check the console.")
        st.stop()

    # Create columns for layout
    col1, col2 = st.columns([7, 3])

    with col1:
        mode = st.radio(
            "Choose your input mode:",
            options=["Live Camera", "Upload Photo", "Take Snapshot"],
            horizontal=True,
            key="input_mode"
        )

        # --- Input Mode Logic ---
        if mode == "Live Camera":
            st.info("Allow webcam access to start real-time food detection. Food detection results update every 2 seconds.")
            
            # Setup for periodic captures
            auto_refresh = st.checkbox("Auto-capture every 2 seconds", value=True)
            
            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.frame_count = 0
                    self.last_processed_time = time.time()
                
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    current_time = time.time()
                    
                    # Increment frame counter
                    self.frame_count += 1
                    st.session_state.frame_counter = self.frame_count
                    
                    # Periodically capture frames for processing
                    if auto_refresh and current_time - self.last_processed_time >= FRAME_CAPTURE_INTERVAL:
                        self.last_processed_time = current_time
                        
                        # Process the frame for detections
                        annotated_frame, detected_classes = process_image_and_detect(img, model, device)
                        
                        # Save results
                        if detected_classes:
                            write_detections_to_file(detected_classes)
                        
                        # Save the processed frame
                        save_frame_to_file(annotated_frame)
                        
                        # Update timestamp in session state
                        st.session_state.last_capture_time = current_time
                    
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(
                key="yolo-food-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=VideoProcessor,
                async_processing=True,
            )
            
            # If video is active, display status and capture info
            if ctx.video_transformer:
                st.caption(f"Frame count: {st.session_state.frame_counter}")
                st.caption(f"Last capture: {datetime.fromtimestamp(st.session_state.last_capture_time).strftime('%H:%M:%S')}")
                
                # Add manual trigger button
                if st.button("Capture and Analyze Now"):
                    st.session_state.refresh_counter += 1
                    st.rerun()
                
                # Show the captured frame if available
                last_frame = load_frame_from_file()
                if last_frame is not None and st.checkbox("Show last processed frame", value=True):
                    st.image(last_frame, caption="Last Processed Frame", channels="BGR", use_container_width=True)
            else:
                if os.path.exists(DETECTION_FILE):
                    # Clean up if no video is active
                    os.remove(DETECTION_FILE)
                if os.path.exists(FRAME_FILE):
                    os.remove(FRAME_FILE)

        elif mode == "Upload Photo":
            uploaded_file = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                annotated_frame, detected_classes = process_image_and_detect(frame, model, device)
                write_detections_to_file(detected_classes)
                save_frame_to_file(annotated_frame)
                st.image(annotated_frame, channels="BGR", caption="Detection Results", use_container_width=True)

        elif mode == "Take Snapshot":
            camera_img = st.camera_input("Take a photo of your food")
            if camera_img is not None:
                pil_image = Image.open(camera_img)
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                annotated_frame, detected_classes = process_image_and_detect(frame, model, device)
                write_detections_to_file(detected_classes)
                save_frame_to_file(annotated_frame)
                st.image(annotated_frame, channels="BGR", caption="Detection Results", use_container_width=True)

    # --- Sidebar for Nutrition Info ---
    with col2:
        st.header("🍽️ Detected Items")
        
        # Read detections from file
        detected_classes = read_detections_from_file()
        
        # Add refresh button in sidebar
        if st.button("Refresh Detections", key="sidebar-refresh"):
            st.rerun()
        
        # Display detected items and nutrition info
        if detected_classes:
            # Calculate and display total calories for current detection
            total_calories, food_calories = calculate_total_calories(detected_classes, nutrition_df)
            if total_calories > 0:
                st.success(f"🔥 Total Calories: {total_calories:.1f} kcal")
            
            st.caption(f"Last updated: {datetime.fromtimestamp(os.path.getmtime(DETECTION_FILE)).strftime('%H:%M:%S')}")
            
            for food, count in detected_classes.items():
                with st.expander(f"{food.title()} ({count})", expanded=True):
                    info = get_nutrition_info(food, nutrition_df)
                    if info:
                        # Calculate calories for this food item
                        item_calories = 0
                        try:
                            if info['Calories'] != 'N/A':
                                item_calories = float(info['Calories']) * count
                        except (ValueError, TypeError):
                            pass
                        
                        st.markdown(f"""
                        **🔥 Calories:** {info['Calories']} kcal per item ({item_calories:.1f} total)  
                        **Protein:** {info['Protein']} g  
                        **Carbs:** {info['Carbohydrates']} g  
                        **Fat:** {info['Fat']} g  
                        **Sugars:** {info['Sugars']} g  
                        **Fiber:** {info['Fiber']} g  
                        
                        ---
                        **Ingredients:** {info['Ingredients']}  
                        [🔗 View Recipe]({info['Recipe']})
                        """)
                    else:
                        st.write("No nutrition information found for this item.")
        else:
            st.write("No food detected yet. Show me what you're eating!")

    st.markdown("---")
    st.success("**Tip:** Point your camera, upload a photo, or take a snapshot. Expand the items in the sidebar for detailed nutrition facts and recipes!")

# Detection History tab
with tab2:
    st.title("📋 Food Detection History")
    
    # Get the detection log data
    logs = get_detection_logs()
    
    if logs:
        # Add filters
        st.subheader("Filter Options")
        col1, col2 = st.columns(2)
        with col1:
            date_filter = st.date_input("Filter by date", datetime.now(), key="logs_date_filter")
        with col2:
            food_filter = st.text_input("Filter by food name", key="logs_food_filter")
        
        # Filter logs
        filtered_logs = []
        for log in logs:
            log_date = datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S").date()
            
            # Apply date filter
            if log_date != date_filter:
                continue
                
            # Apply food filter if provided
            if food_filter and not any(food_filter.lower() in food.lower() for food in log["detections"].keys()):
                continue
                
            filtered_logs.append(log)
        
        # Show logs in a clean table format with cards for each entry
        st.subheader(f"Detection History ({len(filtered_logs)} entries)")
        
        if not filtered_logs:
            st.info("No detection logs match your filter criteria.")
        else:
            # Create a cleaner, more visual history log
            for i, log in enumerate(reversed(filtered_logs)):
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Time:** {log['timestamp'].split()[1]}")
                    
                    with col2:
                        foods_list = ", ".join([f"{food} ({count})" for food, count in log["detections"].items()])
                        st.markdown(f"**Detected:** {foods_list}")
                    
                    with col3:
                        calories = log.get("total_calories", 0)
                        if calories > 0:
                            st.markdown(f"**🔥 {calories:.1f} kcal**")
                        else:
                            st.markdown("**No calorie data**")
                
                if i < len(filtered_logs) - 1:
                    st.divider()
            
            # Add download option for logs
            log_df = pd.DataFrame([
                {
                    "timestamp": log["timestamp"],
                    "foods": ", ".join([f"{food} ({count})" for food, count in log["detections"].items()]),
                    "total_calories": log.get("total_calories", 0)
                }
                for log in filtered_logs
            ])
            
            st.download_button(
                label="Download Logs as CSV",
                data=log_df.to_csv().encode('utf-8'),
                file_name=f"food_detection_logs_{date_filter}.csv",
                mime="text/csv",
            )
    else:
        st.info("No detection logs available yet. Start detecting some food to build your history!")

# Calorie Tracker tab
with tab3:
    st.title("📊 Calorie Intake Tracker")
    
    # Get calorie log data
    calorie_logs = get_calorie_logs()
    
    if calorie_logs:
        # Create date range for analysis
        dates = sorted(calorie_logs.keys())
        
        # Daily overview
        st.subheader("📅 Daily Overview")
        selected_date = st.date_input("Select date to view details", 
                                    datetime.now(), 
                                    key="calorie_date_selector")
        selected_date_str = selected_date.strftime("%Y-%m-%d")
        
        if selected_date_str in calorie_logs:
            day_data = calorie_logs[selected_date_str]
            
            # Display daily summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Calories", f"{day_data['total_calories']:.1f} kcal")
            with col2:
                st.metric("Number of Meals", len(day_data['meals']))
            with col3:
                avg_calories = day_data['total_calories'] / len(day_data['meals']) if day_data['meals'] else 0
                st.metric("Avg per Meal", f"{avg_calories:.1f} kcal")
            
            # Meal breakdown
            st.subheader("🍽️ Meals Breakdown")
            for i, meal in enumerate(day_data['meals']):
                with st.expander(f"Meal {i+1} at {meal['time']} - {meal['calories']:.1f} kcal"):
                    for food, info in meal['foods'].items():
                        st.write(f"• {food.title()}: {info['count']} items × {info['calories_per_item']:.1f} kcal = {info['total_calories']:.1f} kcal")
            
            # Food breakdown pie chart
            if day_data['food_breakdown']:
                st.subheader("🥘 Food Distribution")
                fig = px.pie(
                    values=list(day_data['food_breakdown'].values()),
                    names=[food.title() for food in day_data['food_breakdown'].keys()],
                    title=f"Calorie Distribution for {selected_date_str}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No calorie data available for {selected_date_str}")
        
        # Weekly/Monthly trends
        if len(dates) > 1:
            st.subheader("📈 Calorie Trends")
            
            # Prepare data for plotting
            plot_dates = []
            plot_calories = []
            
            for date in dates:
                plot_dates.append(datetime.strptime(date, "%Y-%m-%d"))
                plot_calories.append(calorie_logs[date]['total_calories'])
            
            # Create trend chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_dates,
                y=plot_calories,
                mode='lines+markers',
                name='Daily Calories',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Daily Calorie Intake Trend",
                xaxis_title="Date",
                yaxis_title="Calories (kcal)",
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly average
            if len(dates) >= 7:
                recent_week = dates[-7:]
                week_total = sum(calorie_logs[date]['total_calories'] for date in recent_week)
                week_avg = week_total / 7
                st.metric("Weekly Average", f"{week_avg:.1f} kcal/day")
        
        # Export calorie data
        st.subheader("📄 Export Data")
        
        # Create comprehensive export data
        export_data = []
        for date, day_data in calorie_logs.items():
            for i, meal in enumerate(day_data['meals']):
                for food, info in meal['foods'].items():
                    export_data.append({
                        'Date': date,
                        'Meal_Number': i + 1,
                        'Time': meal['time'],
                        'Food': food.title(),
                        'Count': info['count'],
                        'Calories_Per_Item': info['calories_per_item'],
                        'Total_Calories': info['total_calories']
                    })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            st.download_button(
                label="Download Detailed Calorie Data as CSV",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"calorie_intake_detailed_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            # Summary export
            summary_data = [
                {
                    'Date': date,
                    'Total_Calories': day_data['total_calories'],
                    'Number_of_Meals': len(day_data['meals']),
                    'Top_Food': max(day_data['food_breakdown'].items(), key=lambda x: x[1])[0].title() if day_data['food_breakdown'] else 'N/A'
                }
                for date, day_data in calorie_logs.items()
            ]
            
            summary_df = pd.DataFrame(summary_data)
            st.download_button(
                label="Download Daily Summary as CSV",
                data=summary_df.to_csv(index=False).encode('utf-8'),
                file_name=f"calorie_intake_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
    else:
        st.info("No calorie tracking data available yet. Start detecting food to build your calorie history!")
        st.markdown("""
        ### How it works:
        1. 🍲 Go to the **Detection** tab and detect some food
        2. 📊 Calorie information will automatically be calculated and stored
        3. 📈 View your daily intake trends and meal breakdowns here
        """)

# Add auto-refresh after a delay if in live mode
if st.session_state.get('input_mode') == "Live Camera":
    time.sleep(2)
    st.rerun()
