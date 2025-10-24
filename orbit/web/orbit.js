import { app } from "/scripts/app.js";

app.registerExtension({
    name: "orbit.modelmerge",
    
    async nodeCreated(node) {
        if (node.comfyClass === "ORBITModelMerge") {
            const origOnDrawForeground = node.onDrawForeground;
            
            node.orbitRotation = 0;
            node.orbitPulse = 0;
            
            node.onDrawForeground = function(ctx) {
                if (origOnDrawForeground) {
                    origOnDrawForeground.apply(this, arguments);
                }
                
                // Animation state
                this.orbitRotation = (this.orbitRotation + 0.02) % (Math.PI * 2);
                this.orbitPulse = (this.orbitPulse + 0.05) % (Math.PI * 2);
                
                // Position in center
                const x = this.size[0] - 125;
                const y = 25;
                const radius = 8;
                
                ctx.save();
                
                // Saturn planet (white sphere)
                const gradient = ctx.createRadialGradient(x - 2, y - 2, 0, x, y, radius);
                gradient.addColorStop(0, '#ffffff');
                gradient.addColorStop(0.7, '#e8e8e8');
                gradient.addColorStop(1, '#d0d0d0');
                
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fill();
                
                // Subtle shadow
                ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
                ctx.beginPath();
                ctx.arc(x + 1, y + 1, radius, 0, Math.PI * 2);
                ctx.fill();
                
                // Animated rings (halo)
                const ringWidth = radius * 2.5;
                const ringHeight = radius * 0.6;
                const pulse = Math.sin(this.orbitPulse) * 0.1 + 0.9;
                
                // Outer ring
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.ellipse(x, y, ringWidth * pulse, ringHeight * pulse, 
                           -this.orbitRotation * 0.5, 0, Math.PI * 2);
                ctx.stroke();
                
                // Inner ring
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.ellipse(x, y, (ringWidth * 0.7) * pulse, (ringHeight * 0.7) * pulse, 
                           -this.orbitRotation * 0.5, 0, Math.PI * 2);
                ctx.stroke();
                
                // Rotating particle/moon
                const moonAngle = this.orbitRotation;
                const moonDist = ringWidth * 0.85 * pulse;
                const moonX = x + Math.cos(moonAngle) * moonDist;
                const moonY = y + Math.sin(moonAngle) * moonDist * (ringHeight / ringWidth);
                
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.beginPath();
                ctx.arc(moonX, moonY, 2, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.restore();
                
                // Request redraw for animation
                if (this.flags.collapsed !== true) {
                    app.canvas.setDirty(true);
                }
            };
            
            // Add tooltip
            node.title = "🪐 ORBIT Merge";
            
            // Style the node with a subtle theme
            node.color = "#1a1a2e";
            node.bgcolor = "#16213e";
            
            // Add informative text to node
            const origGetExtraMenuOptions = node.getExtraMenuOptions;
            node.getExtraMenuOptions = function(_, options) {
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.apply(this, arguments);
                }
                
                options.unshift(
                    {
                        content: "ℹ️ ORBIT Info",
                        callback: () => {
                            alert(
                                "ORBIT - Orthogonal Residual Blend In Tensors\n\n" +
                                "injects orthogonal novelty from Model B into Model A's structure.\n\n" +
                                "• α∥ (parallel): adjustment along A's direction\n" +
                                "• α⊥ (orthogonal): novel features from B\n" +
                                "• trust_k: robustness control (MAD-based)\n" +
                                "• coef_clip: stability clamp for projections\n\n" +
                                "output saved to: ComfyUI/output/"
                            );
                        }
                    },
                    null
                );
            };
        }
    }
});