import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import tempfile
from pathlib import Path
import numpy as np

# Add the monst3r tools to path
CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.append(os.path.join(CURRENT_DIR, 'tools', 'monst3r'))
sys.path.append(os.path.join(CURRENT_DIR, 'tools', 'monst3r', 'demo_data'))

from demo_data.clip import VideoFrameExtractor
from camera_analysis import MonST3RProcessor
from camera_visualization import quaternion_to_euler

# Add GeminiAPI import
sys.path.append(os.path.join(BASE_DIR, 'Tools'))
from gemini_api import GeminiAPI

class CameraControl:
    def __init__(self, weights_path=None, device='cuda', image_size=512, 
                 batch_size=16, output_dir=os.path.join(CURRENT_DIR, 'tools', 'monst3r', 'demo_tmp'), silent=False):
        """
        Initialize CameraControl with MonST3R processor and Gemini API
        """
        # Default Gemini API keys
        # default_gemini_keys = [
        #     "YOUR_GEMINI_API_KEY", 
        #     "YOUR_GEMINI_API_KEY", 
        #     "YOUR_GEMINI_API_KEY"
        # ]
        default_gemini_keys = [
            "YOUR_GEMINI_API_KEY", 
        ]
        
        self.frame_extractor = VideoFrameExtractor()
        self.processor = MonST3RProcessor(
            weights_path=weights_path,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            output_dir=output_dir,
            silent=silent
        )
        self.demo_data_dir = os.path.join(CURRENT_DIR, 'tools', 'monst3r', 'demo_data')
        
        # Initialize Gemini API with default settings
        self.gemini_api = GeminiAPI(
            api_keys=default_gemini_keys, 
            proxy="YOUR_PROXY_URL"
        )

    def analyze_camera_motion_text(self, traj_file, camera_motion_instruction=None):
        """
        Analyze camera motion and generate text report for position and rotation changes
        
        Args:
            traj_file: Path to trajectory file
            camera_motion_instruction: Expected camera motion instruction for scoring
            
        Returns:
            Dictionary containing analysis results and optional score
        """
        # Capture analysis output
        import io
        from contextlib import redirect_stdout
        
        analysis_output = io.StringIO()
        
        with redirect_stdout(analysis_output):
            # Read trajectory data
            data = []
            with open(traj_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    frame = float(parts[0])
                    tx, ty, tz = map(float, parts[1:4])
                    qw, qx, qy, qz = map(float, parts[4:8])
                    data.append([frame, tx, ty, tz, qw, qx, qy, qz])
            
            data = np.array(data)
            
            # Extract position and rotation information
            frames = data[:, 0]
            positions = data[:, 1:4]  # tx, ty, tz
            quaternions = data[:, 4:8]  # qw, qx, qy, qz
            
            # Calculate motion characteristics
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Calculate rotation angles
            euler_angles = []
            for i in range(len(quaternions)):
                qw, qx, qy, qz = quaternions[i]
                roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz)
                euler_angles.append([roll, pitch, yaw])
            
            euler_angles = np.array(euler_angles)
            
            # Calculate angular velocities
            angular_velocities = np.diff(euler_angles, axis=0)
            angular_speeds = np.linalg.norm(angular_velocities, axis=1)
            
            # Generate Camera Position Changes Analysis
            print("\n" + "="*60)
            print("Camera Position Changes Analysis")
            print("="*60)
            
            # Position range analysis
            x_range = positions[:,0].max() - positions[:,0].min()
            y_range = positions[:,1].max() - positions[:,1].min()
            z_range = positions[:,2].max() - positions[:,2].min()
            
            print(f"Position Range Analysis:")
            print(f"   - X-axis (Red, Screen Right): [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}] (range: {x_range:.3f})")
            print(f"   - Y-axis (Green, Scene Depth): [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}] (range: {y_range:.3f})")
            print(f"   - Z-axis (Blue, Screen Down): [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}] (range: {z_range:.3f})")
            
            # Movement direction analysis
            x_movement = positions[-1,0] - positions[0,0]
            y_movement = positions[-1,1] - positions[0,1]
            z_movement = positions[-1,2] - positions[0,2]
            
            print(f"\nOverall Movement Direction:")
            x_dir = "rightward" if x_movement > 0 else "leftward" if x_movement < 0 else "no net movement"
            y_dir = "forward (into scene)" if y_movement > 0 else "backward (out of scene)" if y_movement < 0 else "no net movement"
            z_dir = "downward" if z_movement > 0 else "upward" if z_movement < 0 else "no net movement"
            
            print(f"   - X-axis: {x_dir} ({x_movement:+.3f})")
            print(f"   - Y-axis: {y_dir} ({y_movement:+.3f})")
            print(f"   - Z-axis: {z_dir} ({z_movement:+.3f})")
            
            # Dominant movement axis
            movements = [abs(x_movement), abs(y_movement), abs(z_movement)]
            axis_names = ['X (horizontal)', 'Y (depth)', 'Z (vertical)']
            dominant_axis = axis_names[np.argmax(movements)]
            
            print(f"\nDominant Movement:")
            print(f"   - Primary axis: {dominant_axis}")
            print(f"   - Movement magnitude: {max(movements):.3f}")
            
            # Movement smoothness
            speed_variation = np.std(speeds) / np.mean(speeds) if np.mean(speeds) > 0 else 0
            
            print(f"\nMovement Characteristics:")
            print(f"   - Average speed: {speeds.mean():.4f} units/frame")
            print(f"   - Speed variation: {speed_variation:.3f}")
            
            # Generate Camera Rotation Angles Analysis
            print("\n" + "="*60)
            print("Camera Rotation Angles Analysis")
            print("="*60)
            
            # Rotation range analysis
            roll_range = np.degrees(euler_angles[:,0].max() - euler_angles[:,0].min())
            pitch_range = np.degrees(euler_angles[:,1].max() - euler_angles[:,1].min())
            yaw_range = np.degrees(euler_angles[:,2].max() - euler_angles[:,2].min())
            
            print(f"Rotation Range Analysis:")
            print(f"   - Roll (Red, Lens Axis): [{np.degrees(euler_angles[:,0].min()):.1f}°, {np.degrees(euler_angles[:,0].max()):.1f}°] (range: {roll_range:.1f}°)")
            print(f"   - Pitch (Green, Right Axis): [{np.degrees(euler_angles[:,1].min()):.1f}°, {np.degrees(euler_angles[:,1].max()):.1f}°] (range: {pitch_range:.1f}°)")
            print(f"   - Yaw (Blue, Up Axis): [{np.degrees(euler_angles[:,2].min()):.1f}°, {np.degrees(euler_angles[:,2].max()):.1f}°] (range: {yaw_range:.1f}°)")
            
            # Rotation direction analysis
            roll_change = np.degrees(euler_angles[-1,0] - euler_angles[0,0])
            pitch_change = np.degrees(euler_angles[-1,1] - euler_angles[0,1])
            yaw_change = np.degrees(euler_angles[-1,2] - euler_angles[0,2])
            
            print(f"\nOverall Rotation Changes:")
            roll_desc = "clockwise roll" if roll_change > 0 else "counterclockwise roll" if roll_change < 0 else "no net roll"
            pitch_desc = "downward tilt" if pitch_change > 0 else "upward tilt" if pitch_change < 0 else "no net tilt"
            yaw_desc = "rightward turn" if yaw_change > 0 else "leftward turn" if yaw_change < 0 else "no net turn"
            
            print(f"   - Roll: {roll_desc} ({roll_change:+.1f}°)")
            print(f"   - Pitch: {pitch_desc} ({pitch_change:+.1f}°)")
            print(f"   - Yaw: {yaw_desc} ({yaw_change:+.1f}°)")
            
            # Dominant rotation axis
            rotations = [abs(roll_change), abs(pitch_change), abs(yaw_change)]
            rotation_names = ['Roll (lens rotation)', 'Pitch (vertical tilt)', 'Yaw (horizontal pan)']
            dominant_rotation = rotation_names[np.argmax(rotations)]
            
            print(f"\nDominant Rotation:")
            print(f"   - Primary rotation: {dominant_rotation}")
            print(f"   - Rotation magnitude: {max(rotations):.1f}°")
            
            # Rotation smoothness
            angular_variation = np.std(angular_speeds) / np.mean(angular_speeds) if np.mean(angular_speeds) > 0 else 0
            
            print(f"\nRotation Characteristics:")
            print(f"   - Average angular speed: {angular_speeds.mean():.4f} rad/frame")
            print(f"   - Angular variation: {angular_variation:.3f})")
            
            # Camera behavior interpretation
            print("\n" + "="*60)
            print("Camera Behavior Conclusion:")
            print("="*60 + "\n")
            
            # Determine primary camera movement type
            if max(movements) > 0.05:  # Significant translation
                if dominant_axis == 'X (horizontal)':
                    if x_movement > 0:
                        movement_type = "Right tracking shot"
                    else:
                        movement_type = "Left tracking shot"
                elif dominant_axis == 'Y (depth)':
                    if y_movement > 0:
                        movement_type = "Forward dolly/push-in"
                    else:
                        movement_type = "Backward dolly/pull-out"
                else:  # Z axis
                    if z_movement > 0:
                        movement_type = "Downward crane/tilt"
                    else:
                        movement_type = "Upward crane/lift"
            else:
                movement_type = "Static position"
            
            # Determine primary camera rotation type
            if max(rotations) > 5:  # Significant rotation (>5 degrees)
                if dominant_rotation == 'Roll (lens rotation)':
                    rotation_type = "Dutch angle/roll adjustment"
                elif dominant_rotation == 'Pitch (vertical tilt)':
                    if pitch_change > 0:
                        rotation_type = "Downward tilt"
                    else:
                        rotation_type = "Upward tilt"
                else:  # Yaw
                    if yaw_change > 0:
                        rotation_type = "Rightward pan"
                    else:
                        rotation_type = "Leftward pan"
            else:
                rotation_type = "Minimal rotation"
            
            print(f"   - Primary movement: {movement_type}")
            print(f"   - Primary rotation: {rotation_type}")
            print(f"   - Overall complexity: {'High' if speed_variation > 0.6 or angular_variation > 0.6 else 'Medium' if speed_variation > 0.3 or angular_variation > 0.3 else 'Low'}")
        
        # Get the captured analysis text
        analysis_text = analysis_output.getvalue()
        
        # Print the analysis to console (maintain existing behavior)
        print(analysis_text)
        
        # Score against instruction if provided
        score = None
        score_explanation = None
        if camera_motion_instruction and self.gemini_api:
            score, score_explanation = self._score_camera_motion(
                analysis_text, camera_motion_instruction
            )
        
        return {
            'positions': positions,
            'euler_angles': euler_angles,
            'speeds': speeds,
            'angular_speeds': angular_speeds,
            'movement_type': movement_type,
            'rotation_type': rotation_type,
            'analysis_text': analysis_text,
            'score': score,
            'score_explanation': score_explanation
        }
    
    def _score_camera_motion(self, analysis_text, camera_motion_instruction):
        """Score camera motion against instruction using Gemini API"""
        if not self.gemini_api:
            return None, "Gemini API not initialized"
        
        scoring_prompt = f"""
        You are a professional cinematographer evaluating camera motion quality. Please score how well the actual camera motion matches the intended camera motion instruction.

        CAMERA MOTION INSTRUCTION (What was expected):
        {camera_motion_instruction}

        ACTUAL CAMERA MOTION ANALYSIS (What actually happened):
        {analysis_text}

        SCORING CRITERIA:
        - Score 0: No motion when motion was expected, or motion completely opposite to instruction
        - Score 1: Very poor match - motion exists but wrong direction/type
        - Score 2: Poor match - some similarity but significant differences
        - Score 3: Fair match - generally correct but with notable deviations
        - Score 4: Good match - mostly correct with minor differences
        - Score 5: Excellent match - motion perfectly matches the instruction

        Please provide:
        1. A score from 0-5 (integer only)
        2. A detailed explanation of your scoring reasoning

        Format your response as:
        SCORE: [0-5]
        EXPLANATION: [Your detailed reasoning]
        """
        
        try:
            response = self.gemini_api.generate_from_text(scoring_prompt)
            
            # Parse the response
            lines = response.strip().split('\n')
            score = None
            explanation = ""
            
            for line in lines:
                if line.startswith('SCORE:'):
                    try:
                        score = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('EXPLANATION:'):
                    explanation = line.split(':', 1)[1].strip()
                elif explanation and not line.startswith('SCORE:'):
                    explanation += " " + line.strip()
            
            # Validate score
            if score is None or score < 0 or score > 5:
                return None, f"Invalid score from Gemini API: {response}"
            
            return score, explanation
            
        except Exception as e:
            return None, f"Error calling Gemini API: {str(e)}"

    def process_video(self, video_path, camera_instruction=None):
        """
        Complete pipeline: extract frames from video and process with MonST3R
        
        Args:
            video_path: Path to input video file
            camera_instruction: Expected camera motion instruction for scoring
        
        Returns:
            Tuple of (scene, outfile, imgs, analysis_result) from MonST3R processing
        """
        # Extract sequence name from video path
        seq_name = Path(video_path).stem
        
        # Use default parameters
        fps = 32
        num_frames = 200
        
        # Set up paths
        frames_dir = os.path.join(self.demo_data_dir, seq_name)
        traj_file = os.path.join(self.processor.output_dir, seq_name, 'pred_traj.txt')
        
        print(f"Processing video: {video_path}")
        print(f"Sequence name: {seq_name}")
        print(f"Frames will be saved to: {frames_dir}")
        if camera_instruction:
            print(f"Camera motion instruction: {camera_instruction}")
        
        # Check if trajectory file already exists
        if os.path.exists(traj_file):
            print(f"\n=== Trajectory file already exists: {traj_file} ===")
            print("Skipping processing, performing analysis only...")
            
            # Perform camera motion analysis
            print(f"\n=== Camera Motion Analysis ===")
            analysis_result = self.analyze_camera_motion_text(traj_file, camera_instruction)
            
            # Print score if available
            if analysis_result['score'] is not None:
                print(f"\n=== Camera Motion Score ===")
                print(f"Score: {analysis_result['score']}/5")
                print(f"Explanation: {analysis_result['score_explanation']}")
            
            print(f"\n=== Analysis Complete ===")
            print(f"Results available at: {self.processor.output_dir}/{seq_name}")
            
            return None, traj_file, None, analysis_result
        
        # Step 1: Extract frames from video
        print("\n=== Step 1: Extracting frames ===")
        success = self.frame_extractor.extract_frames_from_video(
            video_path, frames_dir, fps=fps
        )
        
        if not success:
            raise RuntimeError(f"Failed to extract frames from video: {video_path}")
        
        # Step 2: Process frames with MonST3R
        print("\n=== Step 2: Processing with MonST3R ===")
        
        # Default processing parameters
        processing_params = {
            'fps': 0,  # Use all extracted frames
            'num_frames': num_frames,
            'schedule': 'linear',
            'niter': 300,
            'min_conf_thr': 1.1,
            'as_pointcloud': True,
            'mask_sky': False,
            'clean_depth': True,
            'transparent_cams': False,
            'cam_size': 0.05,
            'show_cam': True,
            'scenegraph_type': 'swinstride',
            'winsize': 5,
            'refid': 0,
            'temporal_smoothing_weight': 0.01,
            'translation_weight': '1.0',
            'shared_focal': True,
            'flow_loss_weight': 0.01,
            'flow_loss_start_iter': 0.1,
            'flow_loss_threshold': 25,
            'use_gt_mask': False,
            'real_time': False
        }
        
        # Process the extracted frames
        scene, outfile, imgs = self.processor.process_images(
            frames_dir, seq_name=seq_name, **processing_params
        )
        
        # Add camera motion analysis
        analysis_result = None
        if os.path.exists(traj_file):
            print(f"\n=== Camera Motion Analysis ===")
            analysis_result = self.analyze_camera_motion_text(traj_file, camera_instruction)
            
            # Print score if available
            if analysis_result['score'] is not None:
                print(f"\n=== Camera Motion Score ===")
                print(f"Score: {analysis_result['score']}/5")
                print(f"Explanation: {analysis_result['score_explanation']}")
        
        print(f"\n=== Processing Complete ===")
        print(f"Results saved to: {self.processor.output_dir}/{seq_name}")
        
        return scene, outfile, imgs, analysis_result

    def process_image_directory(self, image_dir, seq_name=None, processing_params=None, 
                              camera_motion_instruction=None):
        """
        Process an existing directory of images with MonST3R
        
        Args:
            image_dir: Directory containing images
            seq_name: Sequence name (if None, will use directory name)
            processing_params: Dict of parameters for MonST3R processing
            camera_motion_instruction: Expected camera motion instruction for scoring
        
        Returns:
            Tuple of (scene, outfile, imgs, analysis_result) from MonST3R processing
        """
        if seq_name is None:
            seq_name = Path(image_dir).name
        
        # Check if trajectory file already exists
        traj_file = os.path.join(self.processor.output_dir, seq_name, 'pred_traj.txt')
        
        print(f"Processing image directory: {image_dir}")
        print(f"Sequence name: {seq_name}")
        if camera_motion_instruction:
            print(f"Camera motion instruction: {camera_motion_instruction}")
        
        if os.path.exists(traj_file):
            print(f"\n=== Trajectory file already exists: {traj_file} ===")
            print("Skipping processing, performing analysis only...")
            
            # Perform camera motion analysis
            print(f"\n=== Camera Motion Analysis ===")
            analysis_result = self.analyze_camera_motion_text(traj_file, camera_motion_instruction)
            
            # Print score if available
            if analysis_result['score'] is not None:
                print(f"\n=== Camera Motion Score ===")
                print(f"Score: {analysis_result['score']}/5")
                print(f"Explanation: {analysis_result['score_explanation']}")
            
            print(f"\n=== Analysis Complete ===")
            print(f"Results available at: {self.processor.output_dir}/{seq_name}")
            
            return None, traj_file, None, analysis_result
        
        # Set default processing parameters
        default_params = {
            'fps': 0,
            'num_frames': 200,
            'schedule': 'linear',
            'niter': 300,
            'min_conf_thr': 1.1,
            'as_pointcloud': True,
            'mask_sky': False,
            'clean_depth': True,
            'transparent_cams': False,
            'cam_size': 0.05,
            'show_cam': True,
            'scenegraph_type': 'swinstride',
            'winsize': 5,
            'refid': 0,
            'temporal_smoothing_weight': 0.01,
            'translation_weight': '1.0',
            'shared_focal': True,
            'flow_loss_weight': 0.01,
            'flow_loss_start_iter': 0.1,
            'flow_loss_threshold': 25,
            'use_gt_mask': False,
            'real_time': False
        }
        
        # Update with user-provided parameters
        if processing_params:
            default_params.update(processing_params)
        
        # Process the images
        scene, outfile, imgs = self.processor.process_images(
            image_dir, seq_name=seq_name, **default_params
        )
        
        # Add camera motion analysis
        analysis_result = None
        if os.path.exists(traj_file):
            print(f"\n=== Camera Motion Analysis ===")
            analysis_result = self.analyze_camera_motion_text(traj_file, camera_motion_instruction)
            
            # Print score if available
            if analysis_result['score'] is not None:
                print(f"\n=== Camera Motion Score ===")
                print(f"Score: {analysis_result['score']}/5")
                print(f"Explanation: {analysis_result['score_explanation']}")
        
        print(f"\n=== Processing Complete ===")
        print(f"Results saved to: {self.processor.output_dir}/{seq_name}")
        
        return scene, outfile, imgs, analysis_result

# Example usage
if __name__ == "__main__":
    # Simple usage - only need video path and camera instruction
    camera_control = CameraControl()
    
    # Process a video with camera motion instruction
    video_path = "your_video_path_here"
    camera_instruction = "your_camera_instruction_here"
    
    scene, outfile, imgs, analysis = camera_control.process_video(
        video_path, 
        camera_instruction
    )
    
    print(f"Processing completed successfully!")
    if analysis and analysis['score'] is not None:
        print(f"Final Score: {analysis['score']}/5")
