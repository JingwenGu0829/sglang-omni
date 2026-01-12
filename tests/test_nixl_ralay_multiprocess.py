# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
import pickle
from queue import Empty

import pytest
import torch

# Set multiprocessing start method to 'spawn' (required for CUDA)
if torch.cuda.is_available():
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

try:
    from sglang_omni.relay.descriptor import Descriptor
    from sglang_omni.relay.nvxl.nixl_connect import RdmaMetadata
except ImportError:
    Descriptor = None
    RdmaMetadata = None


def sender_process(config, queue, done_event, num_transfers, data_size, results):
    """Sender process: creates data and sends via put_async."""

    async def run():
        from sglang_omni.relay.nixl_ralay import NixlRalay

        connector = NixlRalay(config)
        device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else "cpu"

        try:
            for _ in range(num_transfers):
                tensor = torch.randn(data_size, dtype=torch.float32, device=device)
                original = tensor.cpu().clone()

                readable_op = await connector.put_async([Descriptor(tensor)])
                metadata = readable_op.metadata()

                # Serialize metadata
                try:
                    meta_bytes = pickle.dumps(metadata)
                except Exception:
                    meta_dict = (
                        metadata.model_dump()
                        if hasattr(metadata, "model_dump")
                        else metadata.dict()
                    )
                    meta_bytes = pickle.dumps(meta_dict)

                queue.put(
                    {
                        "metadata": meta_bytes,
                        "size": data_size,
                        "dtype": tensor.dtype,
                        "original": pickle.dumps(original),
                    }
                )

                await readable_op.wait_for_completion()

            queue.put(None)
            done_event.wait(timeout=300)
        except Exception as e:
            results["sender_error"] = str(e)
        finally:
            connector.close()

    asyncio.run(run())


def receiver_process(config, queue, done_event, num_transfers, results):
    """Receiver process: receives data via get_async."""

    async def run():
        from sglang_omni.relay.nixl_ralay import NixlRalay
        from sglang_omni.relay.nvxl.nixl_connect import RdmaMetadata

        connector = NixlRalay(config)
        device = f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else "cpu"

        try:
            count = 0
            while count < num_transfers:
                try:
                    item = queue.get(timeout=60)
                    if item is None:
                        break

                    # Deserialize metadata
                    meta_obj = pickle.loads(item["metadata"])
                    metadata = (
                        RdmaMetadata(**meta_obj)
                        if isinstance(meta_obj, dict)
                        else meta_obj
                    )

                    # Receive data
                    buffer = torch.empty(
                        item["size"], dtype=item["dtype"], device=device
                    )
                    read_op = await connector.get_async(metadata, [Descriptor(buffer)])
                    if hasattr(read_op, "wait_for_completion"):
                        await read_op.wait_for_completion()

                    # Verify data
                    original = pickle.loads(item["original"])
                    received = buffer.cpu()
                    assert torch.allclose(
                        original, received, rtol=1e-5, atol=1e-5
                    ), f"Data mismatch in transfer {count + 1}"

                    count += 1
                except Empty:
                    break
                except Exception as e:
                    results["receiver_error"] = str(e)
                    break

            results["transfers_completed"] = count
            done_event.set()
        except Exception as e:
            results["receiver_error"] = str(e)
        finally:
            connector.close()

    asyncio.run(run())


@pytest.mark.skipif(Descriptor is None, reason="Descriptor not available")
def test_multiprocess_transfer_with_nixl_ralay():
    """Test data transfer between two processes using NixlRalay."""
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for this test")

    config0 = {
        "host": "127.0.0.1",
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "device_name": "",
        "gpu_id": 0,
        "worker_id": "worker0",
    }

    config1 = {
        "host": "127.0.0.1",
        "metadata_server": "http://127.0.0.1:8080/metadata",
        "device_name": "",
        "gpu_id": 1 if torch.cuda.is_available() else 0,
        "worker_id": "worker1",
    }

    queue = multiprocessing.Queue()
    done_event = multiprocessing.Event()
    results = multiprocessing.Manager().dict()

    sender = multiprocessing.Process(
        target=sender_process,
        args=(config0, queue, done_event, 5, 100000, results),
    )

    receiver = multiprocessing.Process(
        target=receiver_process,
        args=(config1, queue, done_event, 5, results),
    )

    try:
        sender.start()
        receiver.start()

        sender.join(timeout=300)
        receiver.join(timeout=300)

        if sender.exitcode != 0 or receiver.exitcode != 0:
            pytest.fail(
                f"Process failed: sender={sender.exitcode}, receiver={receiver.exitcode}"
            )

        if "sender_error" in results:
            pytest.fail(f"Sender error: {results['sender_error']}")
        if "receiver_error" in results:
            pytest.fail(f"Receiver error: {results['receiver_error']}")

        assert results.get("transfers_completed", 0) == 5, "Not all transfers completed"

    finally:
        for p in [sender, receiver]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
