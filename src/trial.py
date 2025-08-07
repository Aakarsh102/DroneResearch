def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, scaler=None):
    """FIXED: Train model for one epoch with proper scaler handling"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_dict_accum = {}

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Only debug first batch
            if batch_idx == 0:
                print(f"\n=== DEBUG: First batch of epoch {epoch} ===")
                debug_model_forward(model, batch, device)
                print("=== END DEBUG ===\n")

            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            loss = 0
            loss_dict = {}
            predictions = 0
            
            with torch.cuda.amp.autocast():
                predictions = model(batch, use_teacher_forcing=True)
                
                # Check predictions for NaN/Inf before loss computation
                valid_predictions = True
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        print(f"Warning: Non-finite values in {key}, skipping batch")
                        valid_predictions = False
                        break
                
                if not valid_predictions:
                    # IMPORTANT: Still need to update scaler even when skipping
                    scaler.update()
                    continue

                # Compute loss
                loss, loss_dict = criterion(predictions, batch)
                
                if loss is None or not torch.isfinite(loss):
                    print(f"Warning: Invalid loss in batch {batch_idx}, skipping")
                    scaler.update()
                    continue

                # Skip batch if loss is too high
                if loss > 1000.0:
                    print(f"Skipping batch {batch_idx} due to extremely high loss: {loss}")
                    scaler.update()
                    continue

            # Backward pass
            scaler.scale(loss).backward()

            # Unscale gradients for gradient clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Check if gradients are still too high AFTER clipping
            if total_norm > 10.0:  # Reduced threshold since we're clipping
                print(f"Skipping batch {batch_idx} due to high gradient norm: {total_norm}")
                # CRITICAL: Must call scaler.update() even when skipping
                scaler.update()
                continue

            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()

            # Accumulate statistics
            batch_size = batch['past_positions'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for key, value in loss_dict.items():
                if key not in loss_dict_accum:
                    loss_dict_accum[key] = 0.0
                loss_dict_accum[key] += value * batch_size

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'grad_norm': f"{total_norm:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        except RuntimeError as e:
            print(f"\nError at batch {batch_idx}: {e}")
            # Even on error, update scaler to maintain consistency
            scaler.update()
            continue

    # Average losses
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        for key in loss_dict_accum:
            loss_dict_accum[key] /= total_samples
    else:
        avg_loss = float('inf')
        loss_dict_accum = {'total_loss': float('inf')}

    return avg_loss, loss_dict_accum


# Alternative approach: More robust scaler handling
def train_epoch_robust(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, scaler=None):
    """Alternative: More robust training with better scaler state management"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_dict_accum = {}

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        should_skip_batch = False
        
        try:
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                predictions = model(batch, use_teacher_forcing=True)
                
                # Validate predictions
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        print(f"Warning: Non-finite values in {key}, skipping batch")
                        should_skip_batch = True
                        break
                
                if should_skip_batch:
                    scaler.update()  # Update scaler state
                    continue

                # Compute loss
                loss, loss_dict = criterion(predictions, batch)
                
                if loss is None or not torch.isfinite(loss) or loss > 1000.0:
                    print(f"Warning: Invalid/high loss in batch {batch_idx}: {loss}, skipping")
                    should_skip_batch = True

            if should_skip_batch:
                scaler.update()  # Update scaler state
                continue

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient handling: unscale, clip, check
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step optimizer (scaler.step handles inf/nan gradients automatically)
            scaler.step(optimizer)
            scaler.update()  # Always update after step

            # Only accumulate stats if step was successful
            # Note: scaler.step() may skip the actual optimizer step if gradients are inf/nan
            # but that's handled internally
            batch_size = batch['past_positions'].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for key, value in loss_dict.items():
                if key not in loss_dict_accum:
                    loss_dict_accum[key] = 0.0
                loss_dict_accum[key] += value * batch_size

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'grad_norm': f"{total_norm:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        except RuntimeError as e:
            print(f"\nError at batch {batch_idx}: {e}")
            # On error, still update scaler to maintain state
            scaler.update()
            continue

    # Average losses
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        for key in loss_dict_accum:
            loss_dict_accum[key] /= total_samples
    else:
        avg_loss = float('inf')
        loss_dict_accum = {'total_loss': float('inf')}

    return avg_loss, loss_dict_accum


# Even simpler approach: Let scaler handle everything
def train_epoch_simple(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, scaler=None):
    """Simplest approach: Let scaler handle inf/nan automatically"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_dict_accum = {}

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Standard training step
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(batch, use_teacher_forcing=True)
                loss, loss_dict = criterion(predictions, batch)
                
                # Only skip if loss is None (shouldn't happen with proper criterion)
                if loss is None:
                    continue

            # Let scaler handle inf/nan gradients automatically
            scaler.scale(loss).backward()
            
            # Optional: gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # scaler.step() will skip the update if gradients are inf/nan
            scaler.step(optimizer)
            scaler.update()

            # Only accumulate if loss is finite (for logging purposes)
            if torch.isfinite(loss):
                batch_size = batch['past_positions'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                for key, value in loss_dict.items():
                    if key not in loss_dict_accum:
                        loss_dict_accum[key] = 0.0
                    loss_dict_accum[key] += value * batch_size

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}" if torch.isfinite(loss) else "inf",
                'grad_norm': f"{total_norm:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        except RuntimeError as e:
            print(f"\nError at batch {batch_idx}: {e}")
            continue

    # Average losses
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        for key in loss_dict_accum:
            loss_dict_accum[key] /= total_samples
    else:
        avg_loss = float('inf')
        loss_dict_accum = {'total_loss': float('inf')}

    return avg_loss, loss_dict_accum