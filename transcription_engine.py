#!/usr/bin/env python3
"""
Transcription engine for VoxScribe - Unified transcription method for all STT engines
"""

import os
import time
import gc
import torch
import logging
import asyncio
from pathlib import Path
from audio_processor import process_audio, format_time
from model_loader import load_model

logger = logging.getLogger("voxscribe")


async def transcribe_audio(
    engine, audio_path, model_id, filename=None, file_size=None, save_to_history=True
):
    """Unified transcription method for all STT engines."""

    # Get global manager instances
    from global_managers import get_transcription_logger, get_transcription_manager, get_websocket_manager

    transcription_logger = get_transcription_logger()
    transcription_manager = get_transcription_manager()
    websocket_manager = get_websocket_manager()

    total_start_time = time.time()

    # Extract filename if not provided
    if filename is None:
        filename = Path(audio_path).name

    # Get file size if not provided
    if file_size is None:
        try:
            file_size = Path(audio_path).stat().st_size
        except:
            file_size = 0

    # Generate transcription ID for history
    transcription_id = (
        transcription_manager.generate_transcription_id() if save_to_history else None
    )

    # Log transcription start
    transcription_logger.log_transcription_start(engine, model_id, filename, file_size)
    
    # Send initial progress update
    await websocket_manager.send_transcription_progress(
        engine, model_id, filename, "starting", "Initializing transcription...", 0
    )

    try:
        # Send model loading progress
        await websocket_manager.send_transcription_progress(
            engine, model_id, filename, "loading_model", f"Loading {engine} model...", 10
        )
        
        # Load the appropriate model (this time is not counted in transcription duration)
        model_load_start = time.time()
        model_result = await asyncio.get_event_loop().run_in_executor(
            None, load_model, engine, model_id
        )
        if model_result is None:
            raise Exception("Failed to load model")
        model_load_time = time.time() - model_load_start

        # Send audio processing progress
        await websocket_manager.send_transcription_progress(
            engine, model_id, filename, "processing_audio", "Processing audio file...", 30
        )

        # Process audio (this time is not counted in transcription duration)
        audio_process_start = time.time()
        processed_path, duration_sec = await asyncio.get_event_loop().run_in_executor(
            None, process_audio, audio_path
        )
        if processed_path is None:
            raise Exception("Failed to process audio")
        audio_process_time = time.time() - audio_process_start

        # Send transcription start progress
        await websocket_manager.send_transcription_progress(
            engine, model_id, filename, "transcribing", f"Transcribing with {engine}...", 50
        )

        try:
            # Start measuring actual transcription time
            transcription_start_time = time.time()

            if engine == "whisper":
                model = model_result
                result = model.transcribe(processed_path, word_timestamps=True)

                if not result or "segments" not in result:
                    raise Exception("Whisper transcription failed")

                # Generate CSV with timestamps
                csv_data = [
                    [
                        "From (s)",
                        "To (s)",
                        "From (time)",
                        "To (time)",
                        "Duration",
                        "Transcription",
                    ]
                ]

                for segment in result["segments"]:
                    start_s = segment["start"]
                    end_s = segment["end"]
                    start_formatted = format_time(start_s)
                    end_formatted = format_time(end_s)
                    duration = end_s - start_s

                    csv_data.append(
                        [
                            f"{start_s:.2f}",
                            f"{end_s:.2f}",
                            start_formatted,
                            end_formatted,
                            f"{duration:.2f}",
                            segment["text"].strip(),
                        ]
                    )

            elif engine == "voxtral":
                model, processor = model_result
                inputs = processor.apply_transcription_request(
                    language="en", audio=processed_path, model_id=model_id
                )
                inputs = inputs.to(
                    "cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.bfloat16,
                )

                outputs = model.generate(**inputs, max_new_tokens=500)
                decoded_outputs = processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                transcription_text = decoded_outputs[0]

                # Simple CSV format (no timestamps from Voxtral)
                csv_data = [["Start (s)", "End (s)", "Duration (s)", "Transcription"]]
                csv_data.append(
                    [
                        "0.00",
                        f"{duration_sec:.2f}",
                        f"{duration_sec:.2f}",
                        transcription_text.strip(),
                    ]
                )

            elif engine == "granite":
                model_data = model_result
                model = model_data["model"]
                processor = model_data["processor"]
                tokenizer = model_data["tokenizer"]

                # Load and validate audio
                import torchaudio

                wav, sr = torchaudio.load(processed_path, normalize=True)

                # Ensure mono, 16kHz audio
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != 16000:
                    import torchaudio.transforms as T

                    resampler = T.Resample(sr, 16000)
                    wav = resampler(wav)

                # Create chat prompt for transcription
                system_prompt = "You are Granite a Speech to Text model, developed by IBM. You are a helpful AI assistant"
                user_prompt = (
                    "<|audio|>can you transcribe the speech into a written format?"
                )
                chat = [
                    dict(role="system", content=system_prompt),
                    dict(role="user", content=user_prompt),
                ]
                prompt = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )

                # Process with model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_inputs = processor(
                    prompt, wav, device=device, return_tensors="pt"
                ).to(device)
                model_outputs = model.generate(
                    **model_inputs, max_new_tokens=600, do_sample=False, num_beams=1
                )

                # Extract only the new tokens (response)
                num_input_tokens = model_inputs["input_ids"].shape[-1]
                new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
                output_text = tokenizer.batch_decode(
                    new_tokens, add_special_tokens=False, skip_special_tokens=True
                )

                transcription_text = output_text[0].strip()

                if not transcription_text:
                    raise Exception("Granite transcription failed")

                # Simple CSV format (no detailed timestamps from Granite)
                csv_data = [["Start (s)", "End (s)", "Duration (s)", "Transcription"]]
                csv_data.append(
                    [
                        "0.00",
                        f"{duration_sec:.2f}",
                        f"{duration_sec:.2f}",
                        transcription_text,
                    ]
                )

            elif engine == "nvidia":
                model = model_result

                if "parakeet" in model_id:
                    output = model.transcribe([processed_path], timestamps=True)

                    if not output or len(output) == 0:
                        raise Exception("Parakeet transcription failed")

                    # Generate CSV with timestamps if available
                    csv_data = [
                        [
                            "From (s)",
                            "To (s)",
                            "From (time)",
                            "To (time)",
                            "Duration",
                            "Transcription",
                        ]
                    ]

                    result = output[0]

                    # Check if timestamps are available
                    if hasattr(result, "timestamp") and result.timestamp:
                        segment_timestamps = result.timestamp.get("segment", [])

                        for stamp in segment_timestamps:
                            start_s = stamp["start"]
                            end_s = stamp["end"]
                            start_formatted = format_time(start_s)
                            end_formatted = format_time(end_s)
                            duration = end_s - start_s

                            csv_data.append(
                                [
                                    f"{start_s:.2f}",
                                    f"{end_s:.2f}",
                                    start_formatted,
                                    end_formatted,
                                    f"{duration:.2f}",
                                    stamp["segment"].strip(),
                                ]
                            )
                    else:
                        # Fallback to simple format without timestamps
                        csv_data.append(
                            [
                                "0.00",
                                f"{duration_sec:.2f}",
                                "00:00:00",
                                format_time(duration_sec),
                                f"{duration_sec:.2f}",
                                result.text.strip(),
                            ]
                        )

                elif "canary" in model_id:
                    prompts = [
                        [
                            {
                                "role": "user",
                                "content": f"Transcribe the following: {model.audio_locator_tag}",
                                "audio": [processed_path],
                            }
                        ]
                    ]

                    answer_ids = model.generate(
                        prompts=prompts,
                        max_new_tokens=128,
                    )

                    transcription_text = model.tokenizer.ids_to_text(
                        answer_ids[0].cpu()
                    )

                    if not transcription_text or transcription_text.strip() == "":
                        raise Exception("Canary-Qwen transcription failed")

                    # Simple CSV format (no detailed timestamps from Canary)
                    csv_data = [
                        ["Start (s)", "End (s)", "Duration (s)", "Transcription"]
                    ]
                    csv_data.append(
                        [
                            "0.00",
                            f"{duration_sec:.2f}",
                            f"{duration_sec:.2f}",
                            transcription_text.strip(),
                        ]
                    )

            else:
                raise Exception(f"Unknown engine: {engine}")

            # Calculate actual transcription time (excluding model load and audio processing)
            transcription_time = time.time() - transcription_start_time
            total_processing_time = time.time() - total_start_time

            # Calculate RTFx (Real-Time Factor)
            # RTFx = audio_duration / processing_time
            # Higher RTFx means faster processing (more audio processed per second of processing time)
            rtfx = (
                duration_sec / total_processing_time if total_processing_time > 0 else 0
            )

            # Send completion progress
            await websocket_manager.send_transcription_progress(
                engine, model_id, filename, "complete", "Transcription completed successfully!", 100
            )

            # Log successful transcription
            transcription_logger.log_transcription_complete(
                engine, model_id, filename, total_processing_time, duration_sec, True
            )

            # Send transcription complete notification
            await websocket_manager.send_transcription_complete(
                engine, model_id, filename, True, duration_sec, total_processing_time, rtfx
            )

            # Save to transcription history if requested
            if save_to_history and transcription_id:
                transcription_manager.save_transcription(
                    transcription_id=transcription_id,
                    engine=engine,
                    model_id=model_id,
                    audio_filename=filename,
                    audio_duration=duration_sec,
                    transcription_duration=transcription_time,
                    csv_data=csv_data,
                    success=True,
                    rtfx=rtfx,
                )

            return {
                "success": True,
                "csv_data": csv_data,
                "processing_time": total_processing_time,
                "transcription_time": transcription_time,
                "duration": duration_sec,
                "rtfx": rtfx,
                "transcription_id": transcription_id,
            }

        finally:
            # Cleanup
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        total_processing_time = time.time() - total_start_time
        error_msg = str(e)

        # Send error notification
        await websocket_manager.send_transcription_complete(
            engine, model_id, filename, False, 0, total_processing_time, 0, error_msg
        )

        # Log failed transcription
        transcription_logger.log_transcription_complete(
            engine, model_id, filename, total_processing_time, 0, False, error_msg
        )

        # Save failed transcription to history if requested
        if save_to_history and transcription_id:
            transcription_manager.save_transcription(
                transcription_id=transcription_id,
                engine=engine,
                model_id=model_id,
                audio_filename=filename,
                audio_duration=0,
                transcription_duration=0,
                csv_data=[],
                success=False,
                error=error_msg,
            )

        return {
            "success": False,
            "error": error_msg,
            "transcription_id": transcription_id,
        }
