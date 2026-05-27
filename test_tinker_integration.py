#!/usr/bin/env python3
"""
Test Tinker Integration
=======================
Simple test to verify Tinker API connectivity and basic functionality.
"""

import os
import asyncio
from pathlib import Path

# Create output directories
Path("graphs").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

async def test_tinker_connection():
    """Test basic Tinker API connection"""
    try:
        import tinker
        from tinker import types
        print("Tinker import successful")
        
        # Check API key
        api_key = os.getenv("TINKER_API_KEY")
        if not api_key:
            print("ERROR: TINKER_API_KEY environment variable not set")
            print("Please set it with: export TINKER_API_KEY='your-api-key-here'")
            return False
            
        print("API key found")
        
        # Create service client
        service_client = tinker.ServiceClient(api_key=api_key)
        print("Service client created")
        
        # Test creating a training client (this will test connectivity)
        try:
            print("Testing training client creation...")
            training_client = await service_client.create_lora_training_client_async(
                base_model="meta-llama/Llama-3.1-8B-Instruct",
                rank=16,
            )
            print("Training client created successfully")
            
            # Get tokenizer
            tokenizer = training_client.get_tokenizer()
            print("Tokenizer retrieved successfully")
            
            # Test basic tokenization
            test_text = "Hello, world!"
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            print(f"Tokenization test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
            
            return True
            
        except Exception as e:
            print(f"Error creating training client: {e}")
            return False
            
    except ImportError as e:
        print(f"Tinker import error: {e}")
        return False

async def test_simple_training():
    """Test a very simple training operation"""
    try:
        import tinker
        from tinker import types
        
        api_key = os.getenv("TINKER_API_KEY")
        if not api_key:
            print("No API key - skipping training test")
            return False
            
        service_client = tinker.ServiceClient(api_key=api_key)
        training_client = await service_client.create_lora_training_client_async(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            rank=16,
        )
        tokenizer = training_client.get_tokenizer()
        
        # Create a simple training datum
        test_prompt = "The capital of France is"
        test_completion = " Paris"
        
        # Tokenize using the correct format from tinker-cookbook
        full_text = test_prompt + test_completion
        all_tokens = tokenizer.encode(full_text)
        
        # Input should be right-shifted (remove last token)
        input_tokens = all_tokens[:-1]
        # Target should be left-shifted (remove first token)  
        target_tokens = all_tokens[1:]
        
        # Weights should match target_tokens length exactly
        prompt_len = len(tokenizer.encode(test_prompt))
        # Create weights: 0 for prompt tokens, 1 for completion tokens
        weights = [0]*prompt_len + [1]*(len(target_tokens) - prompt_len)
        
        # Create datum with proper format
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(
                weights=weights,
                target_tokens=target_tokens
            )
        )
        
        print("Created training datum")
        
        # Test forward-backward (this will cost some credits)
        print("Testing forward-backward operation...")
        fwdbwd_future = await training_client.forward_backward_async([datum], "cross_entropy")
        fwdbwd_result = await fwdbwd_future.result_async()
        print(f"Forward-backward successful. Loss: {fwdbwd_result.metrics.get('loss', 'N/A')}")
        
        # Test optimizer step
        print("Testing optimizer step...")
        optim_future = await training_client.optim_step_async(
            types.AdamParams(learning_rate=1e-4)
        )
        await optim_future.result_async()
        print("Optimizer step successful")
        
        return True
        
    except Exception as e:
        print(f"Error in training test: {e}")
        return False

async def test_sampling():
    """Test basic sampling functionality"""
    try:
        import tinker
        from tinker import types
        
        api_key = os.getenv("TINKER_API_KEY")
        if not api_key:
            print("No API key - skipping sampling test")
            return False
            
        service_client = tinker.ServiceClient(api_key=api_key)
        sampling_client = service_client.create_sampling_client(
            base_model="meta-llama/Llama-3.1-8B-Instruct"
        )
        tokenizer = sampling_client.get_tokenizer()
        
        # Test sampling
        prompt = "The best move in tic-tac-toe is"
        prompt_tokens = tokenizer.encode(prompt)
        
        params = types.SamplingParams(max_tokens=20, temperature=0.7)
        result = await sampling_client.sample_async(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            sampling_params=params,
            num_samples=1
        )
        
        generated_tokens = result.sequences[0].tokens
        generated_text = tokenizer.decode(generated_tokens)
        
        print(f"Sampling test successful:")
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"Error in sampling test: {e}")
        return False

async def main():
    """Main test runner"""
    print("=" * 50)
    print("TINKER INTEGRATION TEST")
    print("=" * 50)
    
    # Test basic connection
    print("\n1. Testing Tinker connection...")
    connection_ok = await test_tinker_connection()
    
    if not connection_ok:
        print("\nConnection test failed. Please check your API key and network connection.")
        return
        
    print("\n2. Testing sampling functionality...")
    sampling_ok = await test_sampling()
    
    print("\n3. Testing basic training...")
    training_ok = await test_simple_training()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Connection: {'OK' if connection_ok else 'FAILED'}")
    print(f"Sampling: {'OK' if sampling_ok else 'FAILED'}")
    print(f"Training: {'OK' if training_ok else 'FAILED'}")
    
    if connection_ok and sampling_ok and training_ok:
        print("\nAll tests passed! You can run the full pipeline:")
        print("python main_training_pipeline.py")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
