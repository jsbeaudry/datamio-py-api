# onnx_diarization_clustering.py
import os
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path
from typing import List, Tuple, Dict

class ONNXSpeakerDiarization:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        """Initialize ONNX-based speaker diarization."""
        self.sample_rate = sample_rate
        self.session = ort.InferenceSession(model_path)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✓ Loaded ONNX model: {model_path}")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        if sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = resample(audio, num_samples)
        
        return audio
    
    def process_audio(self, audio: np.ndarray, chunk_duration: float = 10.0) -> np.ndarray:
        """Process audio in chunks and return speaker logits."""
        chunk_samples = int(chunk_duration * self.sample_rate)
        num_chunks = int(np.ceil(len(audio) / chunk_samples))
        
        all_logits = []
        
        print(f"  Processing {num_chunks} chunks...")
        
        for i in range(num_chunks):
            if (i + 1) % 50 == 0:
                print(f"    Chunk {i+1}/{num_chunks}")
            
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, len(audio))
            chunk = audio[start:end]
            
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            input_tensor = chunk[np.newaxis, np.newaxis, :].astype(np.float32)
            
            logits = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )[0]
            
            all_logits.append(logits[0])
        
        return np.concatenate(all_logits, axis=0)
    
    def postprocess_logits(
        self, 
        logits: np.ndarray, 
        threshold: float = 0.25,
        min_duration: float = 0.3,
        frame_duration: float = 0.01696
    ) -> List[Tuple[float, float, int]]:
        """Convert logits to speaker segments."""
        probs = 1 / (1 + np.exp(-logits))
        active = probs > threshold
        
        segments = []
        num_speakers = active.shape[1]
        
        for speaker_id in range(num_speakers):
            speaker_active = active[:, speaker_id]
            
            changes = np.diff(np.concatenate([[False], speaker_active, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            for start, end in zip(starts, ends):
                duration = (end - start) * frame_duration
                
                if duration >= min_duration:
                    start_time = start * frame_duration
                    end_time = end * frame_duration
                    
                    # Calculate mean activation for this segment
                    mean_activation = probs[start:end, speaker_id].mean()
                    
                    segments.append((start_time, end_time, speaker_id, mean_activation))
        
        segments.sort(key=lambda x: x[0])
        
        return segments
    
    def cluster_speakers(
        self, 
        segments: List[Tuple[float, float, int, float]], 
        logits: np.ndarray,
        num_speakers: int = None,
        max_speakers: int = 10
    ) -> Dict[int, int]:
        """
        Cluster model speaker IDs to actual speakers using embeddings.
        
        Args:
            segments: List of (start, end, speaker_id, activation)
            logits: Original logits from model
            num_speakers: Expected number of speakers (None = auto-detect)
            max_speakers: Maximum speakers to consider
        
        Returns:
            Mapping from model speaker_id to clustered speaker_id
        """
        # Get unique speaker IDs from segments
        model_speakers = sorted(set([seg[2] for seg in segments]))
        
        if len(model_speakers) <= 2 and num_speakers is None:
            # If already 2 or fewer, no clustering needed
            return {i: i for i in model_speakers}
        
        print(f"  Clustering {len(model_speakers)} model speakers to {num_speakers or 'auto'} actual speakers...")
        
        # Create speaker embeddings from logits
        probs = 1 / (1 + np.exp(-logits))
        speaker_embeddings = []
        
        for speaker_id in model_speakers:
            # Get all segments for this speaker
            speaker_segments = [seg for seg in segments if seg[2] == speaker_id]
            
            # Calculate weighted average embedding
            total_weight = 0
            embedding = np.zeros(logits.shape[1])
            
            for start_time, end_time, _, activation in speaker_segments:
                start_frame = int(start_time / 0.01696)
                end_frame = int(end_time / 0.01696)
                
                weight = (end_time - start_time) * activation
                embedding += probs[start_frame:end_frame].mean(axis=0) * weight
                total_weight += weight
            
            if total_weight > 0:
                embedding /= total_weight
            
            speaker_embeddings.append(embedding)
        
        speaker_embeddings = np.array(speaker_embeddings)
        
        # Perform hierarchical clustering
        if len(speaker_embeddings) > 1:
            distances = pdist(speaker_embeddings, metric='cosine')
            linkage_matrix = linkage(distances, method='average')
            
            # Determine number of clusters
            if num_speakers is None:
                # Use elbow method or default to 2
                from scipy.cluster.hierarchy import dendrogram, inconsistent
                
                # Try different numbers of clusters
                best_score = float('inf')
                best_n = 2
                
                for n in range(2, min(len(model_speakers), max_speakers) + 1):
                    labels = fcluster(linkage_matrix, n, criterion='maxclust')
                    # Simple silhouette-like score
                    score = inconsistent(linkage_matrix, n).mean()
                    
                    if score < best_score:
                        best_score = score
                        best_n = n
                
                num_speakers = best_n
                print(f"  Auto-detected {num_speakers} speakers")
            
            cluster_labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust')
        else:
            cluster_labels = np.array([1])
        
        # Create mapping
        mapping = {}
        for model_speaker, cluster_id in zip(model_speakers, cluster_labels):
            mapping[model_speaker] = cluster_id - 1  # Make 0-indexed
        
        print(f"  Speaker mapping: {mapping}")
        
        return mapping
    
    def merge_segments(
        self, 
        segments: List[Tuple[float, float, int]], 
        gap_threshold: float = 0.5
    ) -> List[Tuple[float, float, int]]:
        """Merge segments from same speaker with small gaps."""
        if not segments:
            return []
        
        merged = []
        current_start, current_end, current_speaker = segments[0]
        
        for start, end, speaker in segments[1:]:
            if speaker == current_speaker and start - current_end <= gap_threshold:
                current_end = end
            else:
                merged.append((current_start, current_end, current_speaker))
                current_start, current_end, current_speaker = start, end, speaker
        
        merged.append((current_start, current_end, current_speaker))
        
        return merged
    
    def resolve_overlaps(self, segments: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
        """Resolve overlapping segments by keeping the dominant speaker."""
        if not segments:
            return []
        
        resolved = []
        segments = sorted(segments, key=lambda x: x[0])
        
        for i, (start, end, speaker) in enumerate(segments):
            # Check for overlaps with next segments
            while i + 1 < len(segments):
                next_start, next_end, next_speaker = segments[i + 1]
                
                if next_start < end:
                    # Overlap detected
                    if next_speaker == speaker:
                        # Same speaker, merge
                        end = max(end, next_end)
                        i += 1
                    else:
                        # Different speaker, split at midpoint
                        midpoint = (end + next_start) / 2
                        end = midpoint
                        segments[i + 1] = (midpoint, next_end, next_speaker)
                        break
                else:
                    break
            
            if end > start:  # Valid segment
                resolved.append((start, end, speaker))
        
        return resolved
    
    def diarize(
        self, 
        audio_path: str, 
        threshold: float = 0.25,
        min_duration: float = 0.3,
        merge_gap: float = 0.5,
        num_speakers: int = None
    ) -> List[Tuple[float, float, int]]:
        """
        Perform complete speaker diarization with clustering.
        
        Args:
            audio_path: Path to audio file
            threshold: Detection threshold
            min_duration: Minimum segment duration
            merge_gap: Maximum gap to merge segments
            num_speakers: Expected number of speakers (None = auto-detect)
        
        Returns:
            List of (start_time, end_time, speaker_id) tuples
        """
        print(f"Processing: {audio_path}")
        
        # Load audio
        audio = self.load_audio(audio_path)
        print(f"  Audio duration: {len(audio) / self.sample_rate:.2f}s")
        
        # Process
        logits = self.process_audio(audio)
        print(f"  Logits shape: {logits.shape}")
        
        # Extract segments (with activation scores)
        segments_with_activation = self.postprocess_logits(logits, threshold, min_duration)
        print(f"  Found {len(segments_with_activation)} raw segments")
        
        if not segments_with_activation:
            return []
        
        # Cluster speakers
        speaker_mapping = self.cluster_speakers(
            segments_with_activation, 
            logits, 
            num_speakers=num_speakers
        )
        
        # Apply clustering
        segments = [
            (start, end, speaker_mapping[speaker_id])
            for start, end, speaker_id, _ in segments_with_activation
        ]
        
        # Sort by time
        segments.sort(key=lambda x: x[0])
        
        # Merge close segments
        if merge_gap > 0:
            segments = self.merge_segments(segments, merge_gap)
            print(f"  After merging: {len(segments)} segments")
        
        # Resolve overlaps
        segments = self.resolve_overlaps(segments)
        print(f"  After overlap resolution: {len(segments)} segments")
        
        return segments
    
    def save_rttm(self, segments: List[Tuple[float, float, int]], output_path: str, audio_filename: str = "audio"):
        """Save results in RTTM format."""
        with open(output_path, 'w') as f:
            for start, end, speaker in segments:
                duration = end - start
                f.write(f"SPEAKER {audio_filename} 1 {start:.3f} {duration:.3f} <NA> <NA> SPEAKER_{speaker:02d} <NA> <NA>\n")
        print(f"✓ Saved RTTM to {output_path}")

def main():
    MODEL_PATH = "segmentation_model.onnx"
    AUDIO_FILE = "/Users/jeansauvenelbeaudry/Documents/Local-Project/datamio-py-api/input_audio/db65f503-7120-488f-9fd3-64589448dad0.wav"
    
    diarizer = ONNXSpeakerDiarization(MODEL_PATH)
    
    # Specify number of speakers (2 in your case)
    segments = diarizer.diarize(
        AUDIO_FILE, 
        threshold=0.25,
        min_duration=0.3,
        merge_gap=0.5,
        num_speakers=2  # Specify expected speakers
    )
    
    if not segments:
        print("\n⚠️  No segments detected.")
        return
    
    # Display results
    print("\n" + "="*70)
    print("SPEAKER DIARIZATION RESULTS")
    print("="*70 + "\n")
    
    for i, (start, end, speaker) in enumerate(segments[:30]):
        duration = end - start
        print(f"{i+1:3d}. SPEAKER_{speaker} | {start:7.2f}s - {end:7.2f}s | {duration:6.2f}s")
    
    if len(segments) > 30:
        print(f"\n... and {len(segments) - 30} more segments")
    
    # Save RTTM
    audio_name = Path(AUDIO_FILE).stem
    diarizer.save_rttm(segments, f"{audio_name}_diarization.rttm", audio_name)
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70 + "\n")
    
    speakers = set([speaker for _, _, speaker in segments])
    print(f"Total speakers detected: {len(speakers)}\n")
    
    for speaker in sorted(speakers):
        total_time = sum([end - start for start, end, spk in segments if spk == speaker])
        num_segments = len([1 for _, _, spk in segments if spk == speaker])
        percentage = (total_time / (segments[-1][1] - segments[0][0])) * 100
        print(f"SPEAKER_{speaker}: {total_time:7.2f}s ({num_segments:3d} segments) - {percentage:5.1f}%")

if __name__ == "__main__":
    main()