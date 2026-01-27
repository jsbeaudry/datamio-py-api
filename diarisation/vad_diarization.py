# vad_diarization.py
import os
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path
from typing import List, Tuple, Dict

class SileroVAD:
    def __init__(self, sample_rate: int = 16000):
        """Initialize Silero VAD model."""
        self.sample_rate = sample_rate
        
        print("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        self.model = model
        self.get_speech_timestamps = utils[0]
        
        print("✓ Silero VAD loaded")
    
    def detect_speech(
        self, 
        audio: np.ndarray,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30
    ) -> List[Dict[str, int]]:
        """
        Detect speech segments using Silero VAD.
        
        Returns:
            List of dicts with 'start' and 'end' in samples
        """
        audio_tensor = torch.from_numpy(audio).float()
        
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms
        )
        
        return speech_timestamps

class ONNXSpeakerDiarization:
    def __init__(self, model_path: str, sample_rate: int = 16000, use_vad: bool = True):
        """Initialize ONNX-based speaker diarization with optional VAD."""
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        
        # Load diarization model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✓ Loaded ONNX diarization model: {model_path}")
        
        # Load VAD if enabled
        if use_vad:
            self.vad = SileroVAD(sample_rate)
        else:
            self.vad = None
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        if sr != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = resample(audio, num_samples)
        
        return audio
    
    def apply_vad(self, audio: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Apply VAD to filter out non-speech regions.
        
        Returns:
            Filtered audio and list of (start_time, end_time) for speech segments
        """
        if not self.use_vad or self.vad is None:
            duration = len(audio) / self.sample_rate
            return audio, [(0.0, duration)]
        
        print("  Applying Silero VAD...")
        
        speech_timestamps = self.vad.detect_speech(audio)
        
        if not speech_timestamps:
            print("  ⚠️  No speech detected by VAD!")
            return audio, []
        
        # Convert to time segments
        speech_segments = [
            (ts['start'] / self.sample_rate, ts['end'] / self.sample_rate)
            for ts in speech_timestamps
        ]
        
        total_speech = sum([end - start for start, end in speech_segments])
        print(f"  VAD detected {len(speech_segments)} speech segments ({total_speech:.2f}s total)")
        
        return audio, speech_segments
    
    def process_audio(
        self, 
        audio: np.ndarray, 
        speech_segments: List[Tuple[float, float]] = None,
        chunk_duration: float = 10.0
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Process audio in chunks, only on speech regions if VAD is used.
        
        Returns:
            Logits and list of (start_frame, end_frame) for each chunk
        """
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        if speech_segments is None or not self.use_vad:
            # Process entire audio
            num_chunks = int(np.ceil(len(audio) / chunk_samples))
            print(f"  Processing {num_chunks} chunks (no VAD filtering)...")
            
            all_logits = []
            frame_mapping = []
            
            for i in range(num_chunks):
                if (i + 1) % 50 == 0:
                    print(f"    Chunk {i+1}/{num_chunks}")
                
                start = i * chunk_samples
                end = min((i + 1) * chunk_samples, len(audio))
                chunk = audio[start:end]
                
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                
                input_tensor = chunk[np.newaxis, np.newaxis, :].astype(np.float32)
                logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
                
                all_logits.append(logits[0])
                
                # Calculate frame indices
                num_frames = logits[0].shape[0]
                start_frame = len(all_logits) - 1
                frame_mapping.append((i * num_frames, (i + 1) * num_frames))
            
            return np.concatenate(all_logits, axis=0), frame_mapping
        
        else:
            # Process only speech segments
            print(f"  Processing {len(speech_segments)} speech segments...")
            
            all_logits = []
            frame_mapping = []
            global_frame_idx = 0
            
            for seg_idx, (seg_start, seg_end) in enumerate(speech_segments):
                start_sample = int(seg_start * self.sample_rate)
                end_sample = int(seg_end * self.sample_rate)
                segment_audio = audio[start_sample:end_sample]
                
                # Process this segment in chunks
                num_chunks = int(np.ceil(len(segment_audio) / chunk_samples))
                
                for i in range(num_chunks):
                    chunk_start = i * chunk_samples
                    chunk_end = min((i + 1) * chunk_samples, len(segment_audio))
                    chunk = segment_audio[chunk_start:chunk_end]
                    
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                    
                    input_tensor = chunk[np.newaxis, np.newaxis, :].astype(np.float32)
                    logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
                    
                    all_logits.append(logits[0])
                    
                    num_frames = logits[0].shape[0]
                    frame_mapping.append((global_frame_idx, global_frame_idx + num_frames))
                    global_frame_idx += num_frames
            
            return np.concatenate(all_logits, axis=0), frame_mapping
    
    def postprocess_logits(
        self, 
        logits: np.ndarray, 
        threshold: float = 0.25,
        min_duration: float = 0.3,
        frame_duration: float = 0.01696
    ) -> List[Tuple[float, float, int, float]]:
        """Convert logits to speaker segments with activation scores."""
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
                    mean_activation = probs[start:end, speaker_id].mean()
                    
                    segments.append((start_time, end_time, speaker_id, mean_activation))
        
        segments.sort(key=lambda x: x[0])
        return segments
    
    def filter_by_vad(
        self, 
        segments: List[Tuple[float, float, int, float]], 
        speech_segments: List[Tuple[float, float]],
        min_overlap: float = 0.5
    ) -> List[Tuple[float, float, int, float]]:
        """Filter diarization segments to only include those overlapping with VAD speech."""
        if not speech_segments:
            return segments
        
        filtered = []
        
        for start, end, speaker_id, activation in segments:
            duration = end - start
            overlap = 0
            
            # Calculate overlap with speech segments
            for speech_start, speech_end in speech_segments:
                overlap_start = max(start, speech_start)
                overlap_end = min(end, speech_end)
                
                if overlap_end > overlap_start:
                    overlap += overlap_end - overlap_start
            
            # Keep if sufficient overlap
            if overlap / duration >= min_overlap:
                filtered.append((start, end, speaker_id, activation))
        
        print(f"  VAD filtering: {len(segments)} → {len(filtered)} segments")
        return filtered
    
    def cluster_speakers(
        self, 
        segments: List[Tuple[float, float, int, float]], 
        logits: np.ndarray,
        num_speakers: int = None
    ) -> Dict[int, int]:
        """Cluster model speaker IDs to actual speakers."""
        model_speakers = sorted(set([seg[2] for seg in segments]))
        
        if len(model_speakers) <= 2 and num_speakers is None:
            return {i: i for i in model_speakers}
        
        print(f"  Clustering {len(model_speakers)} model speakers to {num_speakers or 'auto'} actual speakers...")
        
        probs = 1 / (1 + np.exp(-logits))
        speaker_embeddings = []
        
        for speaker_id in model_speakers:
            speaker_segments = [seg for seg in segments if seg[2] == speaker_id]
            
            total_weight = 0
            embedding = np.zeros(logits.shape[1])
            
            for start_time, end_time, _, activation in speaker_segments:
                start_frame = int(start_time / 0.01696)
                end_frame = int(end_time / 0.01696)
                
                if end_frame <= len(probs):
                    weight = (end_time - start_time) * activation
                    embedding += probs[start_frame:end_frame].mean(axis=0) * weight
                    total_weight += weight
            
            if total_weight > 0:
                embedding /= total_weight
            
            speaker_embeddings.append(embedding)
        
        speaker_embeddings = np.array(speaker_embeddings)
        
        if len(speaker_embeddings) > 1:
            distances = pdist(speaker_embeddings, metric='cosine')
            linkage_matrix = linkage(distances, method='average')
            
            if num_speakers is None:
                num_speakers = min(2, len(model_speakers))
                print(f"  Auto-detected {num_speakers} speakers")
            
            cluster_labels = fcluster(linkage_matrix, num_speakers, criterion='maxclust')
        else:
            cluster_labels = np.array([1])
        
        mapping = {model_speaker: cluster_id - 1 for model_speaker, cluster_id in zip(model_speakers, cluster_labels)}
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
    
    def diarize(
        self, 
        audio_path: str, 
        threshold: float = 0.25,
        min_duration: float = 0.3,
        merge_gap: float = 0.5,
        num_speakers: int = None,
        vad_threshold: float = 0.5
    ) -> List[Tuple[float, float, int]]:
        """
        Perform complete speaker diarization with VAD filtering.
        
        Args:
            audio_path: Path to audio file
            threshold: Diarization detection threshold
            min_duration: Minimum segment duration
            merge_gap: Maximum gap to merge segments
            num_speakers: Expected number of speakers (None = auto-detect)
            vad_threshold: VAD detection threshold
        
        Returns:
            List of (start_time, end_time, speaker_id) tuples
        """
        print(f"Processing: {audio_path}")
        
        # Load audio
        audio = self.load_audio(audio_path)
        print(f"  Audio duration: {len(audio) / self.sample_rate:.2f}s")
        
        # Apply VAD
        audio, speech_segments = self.apply_vad(audio)
        
        if not speech_segments and self.use_vad:
            print("  No speech detected!")
            return []
        
        # Process audio
        logits, _ = self.process_audio(audio, speech_segments if self.use_vad else None)
        print(f"  Logits shape: {logits.shape}")
        
        # Extract segments
        segments_with_activation = self.postprocess_logits(logits, threshold, min_duration)
        print(f"  Found {len(segments_with_activation)} raw segments")
        
        if not segments_with_activation:
            return []
        
        # Filter by VAD
        if self.use_vad and speech_segments:
            segments_with_activation = self.filter_by_vad(segments_with_activation, speech_segments)
        
        # Cluster speakers
        speaker_mapping = self.cluster_speakers(segments_with_activation, logits, num_speakers)
        
        # Apply clustering
        segments = [
            (start, end, speaker_mapping[speaker_id])
            for start, end, speaker_id, _ in segments_with_activation
        ]
        
        segments.sort(key=lambda x: x[0])
        
        # Merge
        if merge_gap > 0:
            segments = self.merge_segments(segments, merge_gap)
            print(f"  After merging: {len(segments)} segments")
        
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
    
    # Initialize with VAD enabled
    diarizer = ONNXSpeakerDiarization(MODEL_PATH, use_vad=True)
    
    # Perform diarization
    segments = diarizer.diarize(
        AUDIO_FILE, 
        threshold=0.25,
        min_duration=0.3,
        merge_gap=0.5,
        num_speakers=2,
        vad_threshold=0.5
    )
    
    if not segments:
        print("\n⚠️  No segments detected.")
        return
    
    # Display results
    print("\n" + "="*70)
    print("SPEAKER DIARIZATION RESULTS (WITH VAD)")
    print("="*70 + "\n")
    
    for i, (start, end, speaker) in enumerate(segments[:30]):
        duration = end - start
        print(f"{i+1:3d}. SPEAKER_{speaker} | {start:7.2f}s - {end:7.2f}s | {duration:6.2f}s")
    
    if len(segments) > 30:
        print(f"\n... and {len(segments) - 30} more segments")
    
    # Save RTTM
    audio_name = Path(AUDIO_FILE).stem
    diarizer.save_rttm(segments, f"{audio_name}_vad_diarization.rttm", audio_name)
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70 + "\n")
    
    speakers = set([speaker for _, _, speaker in segments])
    print(f"Total speakers detected: {len(speakers)}\n")
    
    total_duration = segments[-1][1] if segments else 0
    
    for speaker in sorted(speakers):
        total_time = sum([end - start for start, end, spk in segments if spk == speaker])
        num_segments = len([1 for _, _, spk in segments if spk == speaker])
        percentage = (total_time / total_duration * 100) if total_duration > 0 else 0
        print(f"SPEAKER_{speaker}: {total_time:7.2f}s ({num_segments:3d} segments) - {percentage:5.1f}%")

if __name__ == "__main__":
    main()