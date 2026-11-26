package org.offirst.offline_dr_screening

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import android.text.Layout
import android.text.StaticLayout
import android.text.TextPaint
import android.text.TextUtils
import io.flutter.plugin.common.MethodCall
import java.io.File

internal object ReportPdf {
    private const val PAGE_W = 595f
    private const val PAGE_H = 842f
    private const val MARGIN = 42.5f

    private val ink = Color.parseColor("#0F172A")
    private val slate = Color.parseColor("#475569")
    private val labelInk = Color.parseColor("#64748B")
    private val metaInk = Color.parseColor("#334155")
    private val lineColor = Color.parseColor("#E2E8F0")
    private val cardFill = Color.parseColor("#F8FAFC")
    private val hairline = Color.parseColor("#CBD5E1")
    private val photoWell = Color.parseColor("#020617")

    private class Tone(
        val solid: Int,
        val tint: Int,
        val ink: Int,
        val banner: Int,
        val edge: Int,
    )

    private fun tone(level: String?): Tone = when (level) {
        "mild" -> Tone(
            Color.parseColor("#EAB308"),
            Color.parseColor("#FEF9C3"),
            Color.parseColor("#854D0E"),
            Color.parseColor("#FEFCE8"),
            Color.parseColor("#FDE047"),
        )
        "moderate" -> Tone(
            Color.parseColor("#F97316"),
            Color.parseColor("#FFEDD5"),
            Color.parseColor("#9A3412"),
            Color.parseColor("#FFF7ED"),
            Color.parseColor("#FDBA74"),
        )
        "severe" -> Tone(
            Color.parseColor("#EF4444"),
            Color.parseColor("#FEE2E2"),
            Color.parseColor("#991B1B"),
            Color.parseColor("#FEF2F2"),
            Color.parseColor("#FCA5A5"),
        )
        "unavailable" -> Tone(
            Color.parseColor("#94A3B8"),
            Color.parseColor("#F1F5F9"),
            Color.parseColor("#475569"),
            Color.parseColor("#F8FAFC"),
            Color.parseColor("#CBD5E1"),
        )
        else -> Tone(
            Color.parseColor("#22C55E"),
            Color.parseColor("#DCFCE7"),
            Color.parseColor("#166534"),
            Color.parseColor("#F0FDF4"),
            Color.parseColor("#86EFAC"),
        )
    }

    fun write(context: Context, call: MethodCall): File {
        val visit = requireNotNull(call.argument<String>("visit"))
        require(visit.matches(Regex("[A-Za-z0-9-]{1,120}")))
        val folder = File(context.filesDir, "screening_reports").apply { mkdirs() }
        val file = File(folder, "OFFIRST-$visit-v6.pdf")
        if (file.isFile) return file
        val reportId = call.argument<String>("reportId") ?: "OFFIRST-$visit"
        val dateTime = call.argument<String>("dateTime").orEmpty()
        val campSite = shown(call.argument<String>("campSite"))
        val disclaimer = call.argument<String>("disclaimer")
            ?: "AI-assisted screening summary for triage only — not a standalone medical diagnosis. Consult a qualified ophthalmologist."
        val patient = call.argument<Map<String, Any?>>("patient") ?: emptyMap()
        val triage = call.argument<Map<String, Any?>>("triage") ?: emptyMap()
        val drEyes = mapList(call.argument<Any>("drEyes"))
        val findings = mapList(call.argument<Any>("findings"))
        val photos = mapList(call.argument<Any>("photos"))
        val findingsState = call.argument<String>("findingsState") ?: "unavailable"
        val notFlagged = call.argument<String>("notFlagged").orEmpty()
        val staging = File.createTempFile("report-", ".staged", folder)
        try {
            val pdf = PdfDocument()
            try {
                drawSummary(
                    pdf, reportId, dateTime, campSite, disclaimer, patient, triage,
                    drEyes, findings, findingsState, notFlagged,
                )
                drawPhotos(pdf, reportId, dateTime, patient, photos)
                staging.outputStream().use { pdf.writeTo(it) }
            } finally {
                pdf.close()
            }
            check(staging.renameTo(file))
            return file
        } finally {
            staging.delete()
        }
    }

    private fun drawSummary(
        pdf: PdfDocument,
        reportId: String,
        dateTime: String,
        campSite: String,
        disclaimer: String,
        patient: Map<String, Any?>,
        triage: Map<String, Any?>,
        drEyes: List<Map<String, Any?>>,
        findings: List<Map<String, Any?>>,
        findingsState: String,
        notFlagged: String,
    ) {
        val page = pdf.startPage(PdfDocument.PageInfo.Builder(PAGE_W.toInt(), PAGE_H.toInt(), 1).create())
        val canvas = page.canvas
        var y = 40f
        y = drawHeader(canvas, y, reportId, dateTime, campSite)
        y = drawPatient(canvas, y, patient)
        y = drawTriage(canvas, y, triage)
        y = drawDr(canvas, y, drEyes)
        val legendTop = PAGE_H - 176f
        drawFindings(canvas, y, legendTop - 8f, findings, findingsState, notFlagged)
        drawLegend(canvas, legendTop)
        drawSignoff(canvas, PAGE_H - 78f)
        drawFooter(canvas, disclaimer, "Page 1 of 2", PAGE_H - 22f)
        pdf.finishPage(page)
    }

    private fun drawHeader(
        canvas: Canvas,
        top: Float,
        reportId: String,
        dateTime: String,
        campSite: String,
    ): Float {
        val title = textPaint(15f, ink, true)
        canvas.drawText("OFFIRST Retinal Screening Report", MARGIN, top + 14f, title)
        canvas.drawText(
            "Offline AI-Assisted Fundus Triage & Multi-Disease Screening Summary",
            MARGIN,
            top + 28f,
            textPaint(8.25f, slate, false),
        )
        val idPaint = textPaint(9f, ink, true)
        val meta = textPaint(8.25f, metaInk, false)
        val id = fit("Report ID: $reportId", idPaint, 210f)
        val date = fit("Date: $dateTime", meta, 210f)
        val camp = fit("Camp Site: $campSite", meta, 210f)
        val right = PAGE_W - MARGIN
        canvas.drawText(id, right - idPaint.measureText(id), top + 10f, idPaint)
        canvas.drawText(date, right - meta.measureText(date), top + 22f, meta)
        canvas.drawText(camp, right - meta.measureText(camp), top + 34f, meta)
        canvas.drawRect(MARGIN, top + 42f, right, top + 44f, Paint().apply { color = ink })
        return top + 54f
    }

    private fun drawPatient(canvas: Canvas, top: Float, patient: Map<String, Any?>): Float {
        val height = 78f
        val rect = RectF(MARGIN, top, PAGE_W - MARGIN, top + height)
        card(canvas, rect, cardFill)
        canvas.drawText("PATIENT DETAILS", MARGIN + 10f, top + 14f, textPaint(7.5f, labelInk, true))
        val fields = listOf(
            "Patient Name" to shown(patient["name"] as? String),
            "Patient ID" to shown(patient["id"] as? String),
            "Age / Sex" to shown(patient["ageSex"] as? String),
            "Contact" to shown(patient["contact"] as? String),
            "Known Diabetes" to shown(patient["diabetes"] as? String),
            "Eyes Captured" to shown(patient["eyes"] as? String),
            "Operator" to shown(patient["operator"] as? String),
            "Capture Device" to shown(patient["device"] as? String),
        )
        val colW = (rect.width() - 20f) / 4f
        val valuePaint = textPaint(9.4f, ink, true)
        val keyPaint = textPaint(7.5f, labelInk, false)
        fields.forEachIndexed { index, (key, value) ->
            val x = MARGIN + 10f + (index % 4) * colW
            val y = top + 30f + (index / 4) * 26f
            canvas.drawText(key, x, y, keyPaint)
            canvas.drawText(fit(value, valuePaint, colW - 8f), x, y + 12f, valuePaint)
        }
        return top + height + 8f
    }

    private fun drawTriage(canvas: Canvas, top: Float, triage: Map<String, Any?>): Float {
        val level = triage["level"] as? String ?: "unavailable"
        val tone = tone(level)
        val title = triage["title"] as? String ?: "Screening Grade Unavailable"
        val subtitle = triage["subtitle"] as? String ?: ""
        val badge = triage["badge"] as? String ?: "NOT GRADED"
        val badgePaint = textPaint(8.25f, if (level == "mild") tone.ink else Color.WHITE, true)
        val badgeW = badgePaint.measureText(badge) + 16f
        val badgeLeft = PAGE_W - MARGIN - 10f - badgeW
        val subPaint = textPaint(8.5f, tone.ink, false)
        val sub = staticLayout(subtitle, subPaint, (badgeLeft - MARGIN - 20f).toInt().coerceAtLeast(1), 2)
        val height = (28f + sub.height).coerceAtLeast(48f)
        val rect = RectF(MARGIN, top, PAGE_W - MARGIN, top + height)
        canvas.drawRoundRect(rect, 8f, 8f, fill(tone.banner))
        canvas.drawRoundRect(rect, 8f, 8f, stroke(tone.edge, 1.2f))
        canvas.drawRoundRect(
            RectF(MARGIN + 1f, top + 8f, MARGIN + 5.5f, top + height - 8f),
            2f, 2f, fill(tone.solid),
        )
        val titlePaint = textPaint(10.5f, tone.ink, true)
        canvas.drawText(fit(title, titlePaint, badgeLeft - MARGIN - 18f), MARGIN + 12f, top + 16f, titlePaint)
        canvas.save()
        canvas.translate(MARGIN + 12f, top + 20f)
        sub.draw(canvas)
        canvas.restore()
        val badgeTop = top + (height - 16f) / 2f
        val badgeRect = RectF(badgeLeft, badgeTop, badgeLeft + badgeW, badgeTop + 16f)
        canvas.drawRoundRect(badgeRect, 8f, 8f, fill(tone.solid))
        canvas.drawText(badge, badgeLeft + 8f, centerText(badgePaint, badgeTop, 16f), badgePaint)
        return top + height + 8f
    }

    private fun drawDr(canvas: Canvas, top: Float, eyes: List<Map<String, Any?>>): Float {
        val rows = if (eyes.isEmpty()) 1 else eyes.size
        val height = 24f + rows * 30f
        val rect = RectF(MARGIN, top, PAGE_W - MARGIN, top + height)
        card(canvas, rect, Color.WHITE)
        canvas.drawText(
            "DIABETIC RETINOPATHY (DR) SEVERITY INDEX",
            MARGIN + 10f, top + 14f, textPaint(7.5f, labelInk, true),
        )
        if (eyes.isEmpty()) {
            canvas.drawText(
                "No eye images captured.",
                MARGIN + 10f, top + 34f, textPaint(9f, slate, false),
            )
            return top + height + 8f
        }
        val labels = listOf("Healthy", "Mild", "Moderate", "Severe")
        val levels = listOf("healthy", "mild", "moderate", "severe")
        eyes.forEachIndexed { index, eye ->
            val row = top + 22f + index * 30f
            val name = eye["name"] as? String ?: ""
            val caption = eye["caption"] as? String ?: ""
            canvas.drawText(name, MARGIN + 10f, row + 12f, textPaint(9f, ink, true))
            canvas.drawText(caption, MARGIN + 10f, row + 23f, textPaint(7f, labelInk, false))
            val active = (eye["gradeIndex"] as? Number)?.toInt() ?: -1
            val trackLeft = MARGIN + 108f
            val trackRight = PAGE_W - MARGIN - 78f
            val gap = 4f
            val segW = (trackRight - trackLeft - gap * 3f) / 4f
            for (segment in 0..3) {
                val tone = tone(levels[segment])
                val x = trackLeft + segment * (segW + gap)
                val box = RectF(x, row + 4f, x + segW, row + 22f)
                val on = segment == active
                canvas.drawRoundRect(box, 4f, 4f, fill(if (on) tone.solid else tone.tint))
                if (on) canvas.drawRoundRect(box, 4f, 4f, stroke(ink, 1.1f))
                val textColor = if (on && levels[segment] != "mild") Color.WHITE else tone.ink
                val paint = textPaint(7f, textColor, true)
                val word = fit(labels[segment], paint, segW - 4f)
                canvas.drawText(word, x + (segW - paint.measureText(word)) / 2f, centerText(paint, box.top, box.height()), paint)
            }
            val score = eye["score"] as? String ?: "—"
            val scorePaint = textPaint(9f, tone(eye["level"] as? String).ink, true)
            val shown = fit(score, scorePaint, 70f)
            canvas.drawText(shown, PAGE_W - MARGIN - 8f - scorePaint.measureText(shown), row + 16f, scorePaint)
        }
        return top + height + 8f
    }

    private fun drawFindings(
        canvas: Canvas,
        top: Float,
        bottom: Float,
        findings: List<Map<String, Any?>>,
        state: String,
        notFlagged: String,
    ) {
        if (bottom - top < 36f) return
        val rect = RectF(MARGIN, top, PAGE_W - MARGIN, bottom)
        card(canvas, rect, Color.WHITE)
        canvas.drawText(
            "OTHER SCREENING FINDINGS (MULTI-DISEASE)",
            MARGIN + 10f, top + 14f, textPaint(7.5f, labelInk, true),
        )
        if (state == "clear") {
            val bar = RectF(MARGIN + 10f, top + 24f, PAGE_W - MARGIN - 10f, top + 42f)
            val tone = tone("healthy")
            canvas.drawRoundRect(bar, 6f, 6f, fill(tone.tint))
            canvas.drawCircle(MARGIN + 22f, bar.centerY(), 3.2f, fill(tone.solid))
            canvas.drawText(
                "No additional screening flags detected",
                MARGIN + 30f,
                centerText(textPaint(8.5f, tone.ink, true), bar.top, bar.height()),
                textPaint(8.5f, tone.ink, true),
            )
            return
        }
        if (state != "flags") {
            canvas.drawText(
                "Multi-disease grade unavailable for this visit.",
                MARGIN + 10f, top + 34f, textPaint(8.5f, slate, false),
            )
            return
        }
        val tableTop = top + 22f
        val noteHeight = if (notFlagged.isBlank()) 0f else 28f
        val headerH = 14f
        val rowCount = findings.size.coerceAtLeast(1)
        val rowH = ((bottom - 8f - noteHeight - tableTop - headerH) / rowCount).coerceAtMost(18f)
        val left = MARGIN + 8f
        val right = PAGE_W - MARGIN - 8f
        val width = right - left
        val nameW = width * 0.42f
        val eyeW = width * 0.16f
        val signalW = width * 0.26f
        val header = textPaint(7f, labelInk, true)
        canvas.drawText("CONDITION", left, tableTop + 10f, header)
        canvas.drawText("EYE", left + nameW, tableTop + 10f, header)
        canvas.drawText("SEVERITY / SIGNAL", left + nameW + eyeW, tableTop + 10f, header)
        val confidence = "CONFIDENCE"
        canvas.drawText(confidence, right - header.measureText(confidence), tableTop + 10f, header)
        canvas.drawRect(left, tableTop + 13f, right, tableTop + 14.2f, fill(lineColor))
        val namePaint = textPaint(8f, ink, true)
        val eyePaint = textPaint(8f, metaInk, false)
        findings.forEachIndexed { index, row ->
            val y = tableTop + headerH + index * rowH
            if (y + rowH > bottom - noteHeight - 4f) return@forEachIndexed
            canvas.drawText(fit(row["name"] as? String ?: "", namePaint, nameW - 8f), left, y + rowH * 0.65f, namePaint)
            canvas.drawText(fit(row["eye"] as? String ?: "", eyePaint, eyeW - 6f), left + nameW, y + rowH * 0.65f, eyePaint)
            drawPill(
                canvas,
                left + nameW + eyeW,
                y + (rowH - 13f) / 2f,
                13f,
                row["signal"] as? String ?: "",
                row["level"] as? String ?: "mild",
                signalW - 6f,
            )
            val score = row["confidence"] as? String ?: ""
            val scorePaint = textPaint(8f, ink, true)
            canvas.drawText(score, right - scorePaint.measureText(score), y + rowH * 0.65f, scorePaint)
            canvas.drawRect(left, y + rowH - 0.6f, right, y + rowH, fill(Color.parseColor("#F1F5F9")))
        }
        if (notFlagged.isNotBlank()) {
            val noteTop = bottom - noteHeight
            canvas.drawLine(
                left, noteTop, right, noteTop,
                Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = lineColor
                    strokeWidth = 1f
                    style = Paint.Style.STROKE
                    pathEffect = DashPathEffect(floatArrayOf(2.5f, 2.5f), 0f)
                },
            )
            val lead = textPaint(7.5f, tone("healthy").ink, true)
            canvas.drawText("Not Flagged (Low Risk): ", left, noteTop + 12f, lead)
            val rest = textPaint(7.5f, slate, false)
            val prefix = lead.measureText("Not Flagged (Low Risk): ")
            val layout = staticLayout(notFlagged, rest, (right - left - prefix).toInt().coerceAtLeast(1), 2)
            canvas.save()
            canvas.translate(left + prefix, noteTop + 4f)
            layout.draw(canvas)
            canvas.restore()
        }
    }

    private fun drawLegend(canvas: Canvas, top: Float) {
        val height = 80f
        val rect = RectF(MARGIN, top, PAGE_W - MARGIN, top + height)
        card(canvas, rect, cardFill)
        canvas.drawText(
            "TIP: HOW TO READ YOUR COLOR-CODED RESULT",
            MARGIN + 10f, top + 13f, textPaint(7.5f, labelInk, true),
        )
        val items = listOf(
            Triple("healthy", "Green · Healthy", "No significant signs detected. Continue annual routine eye screening."),
            Triple("mild", "Yellow · Mild / Borderline", "Early or borderline signs. Follow up with an optometrist or doctor within 3–6 months."),
            Triple("moderate", "Orange · Moderate", "Referable changes detected. Schedule an ophthalmologist exam within 1 month."),
            Triple("severe", "Red · Severe / High", "High-risk or vision-threatening signs. Seek urgent specialist care within 1–2 weeks."),
        )
        val gap = 6f
        val cardW = (rect.width() - 20f - gap * 3f) / 4f
        val headPaint = textPaint(7.4f, ink, true)
        val bodyPaint = textPaint(6.8f, slate, false)
        items.forEachIndexed { index, item ->
            val x = MARGIN + 10f + index * (cardW + gap)
            val box = RectF(x, top + 20f, x + cardW, top + height - 6f)
            card(canvas, box, Color.WHITE)
            val tone = tone(item.first)
            canvas.drawRoundRect(RectF(x + 6f, top + 26f, x + 15f, top + 35f), 2f, 2f, fill(tone.solid))
            val head = staticLayout(item.second, headPaint, (cardW - 22f).toInt(), 2)
            canvas.save()
            canvas.translate(x + 18f, top + 24f)
            head.draw(canvas)
            canvas.restore()
            val body = staticLayout(item.third, bodyPaint, (cardW - 12f).toInt(), 4)
            canvas.save()
            canvas.translate(x + 6f, top + 26f + head.height)
            body.draw(canvas)
            canvas.restore()
        }
    }

    private fun drawSignoff(canvas: Canvas, top: Float) {
        val rect = RectF(MARGIN, top, PAGE_W - MARGIN, top + 34f)
        card(canvas, rect, Color.WHITE)
        val labels = listOf(
            "Clinician Confirmation (Optional):",
            "Confirmed Grade:",
            "Doctor Signature:",
        )
        val weights = listOf(1.4f, 1f, 1f)
        val inner = rect.width() - 16f
        val unit = inner / weights.sum()
        val paint = textPaint(7.5f, slate, true)
        var x = MARGIN + 8f
        labels.forEachIndexed { index, label ->
            val colW = unit * weights[index]
            canvas.drawText(fit(label, paint, colW - 10f), x, top + 12f, paint)
            canvas.drawRect(x, top + 26f, x + colW - 12f, top + 27f, fill(hairline))
            x += colW
        }
    }

    private fun drawPhotos(
        pdf: PdfDocument,
        reportId: String,
        dateTime: String,
        patient: Map<String, Any?>,
        photos: List<Map<String, Any?>>,
    ) {
        val page = pdf.startPage(PdfDocument.PageInfo.Builder(PAGE_W.toInt(), PAGE_H.toInt(), 2).create())
        val canvas = page.canvas
        val name = shown(patient["name"] as? String)
        val id = shown(patient["id"] as? String)
        var y = 40f
        canvas.drawText("Retinal Fundus Photographs", MARGIN, y + 12f, textPaint(12.75f, ink, true))
        canvas.drawText(
            fit("Patient: $name ($id) · Captured $dateTime", textPaint(8f, slate, false), 330f),
            MARGIN, y + 26f, textPaint(8f, slate, false),
        )
        val idPaint = textPaint(8f, ink, true)
        val idLine = fit("Report ID: $reportId", idPaint, 200f)
        val right = PAGE_W - MARGIN
        canvas.drawText(idLine, right - idPaint.measureText(idLine), y + 10f, idPaint)
        val view = textPaint(8f, metaInk, false)
        val viewLine = "Vertical Eye View (OD / OS)"
        canvas.drawText(viewLine, right - view.measureText(viewLine), y + 22f, view)
        canvas.drawRect(MARGIN, y + 34f, right, y + 36f, fill(ink))
        y += 46f
        val footerTop = PAGE_H - 28f
        if (photos.isEmpty()) {
            val rect = RectF(MARGIN, y, right, y + 80f)
            card(canvas, rect, cardFill)
            canvas.drawText(
                "No fundus images were saved with this visit.",
                MARGIN + 12f, y + 36f, textPaint(10f, slate, false),
            )
        } else {
            val gap = 8f
            val cardH = (footerTop - y - gap * (photos.size - 1)) / photos.size
            photos.forEachIndexed { index, photo ->
                val cardTop = y + index * (cardH + gap)
                drawPhotoCard(canvas, RectF(MARGIN, cardTop, right, cardTop + cardH), photo)
            }
        }
        drawFooter(
            canvas,
            "OFFIRST Offline Retinal Screening · Images shown for clinical reference alongside Page 1 triage summary.",
            "Page 2 of 2",
            PAGE_H - 18f,
        )
        pdf.finishPage(page)
    }

    private fun drawPhotoCard(canvas: Canvas, rect: RectF, photo: Map<String, Any?>) {
        card(canvas, rect, cardFill)
        val title = photo["title"] as? String ?: ""
        canvas.drawText(title, rect.left + 10f, rect.top + 16f, textPaint(11f, ink, true))
        val pills = mapList(photo["pills"]).take(5)
        val sized = pills.map { pill ->
            val label = pill["label"] as? String ?: ""
            val level = pill["level"] as? String ?: "healthy"
            val paint = textPaint(7f, tone(level).ink, true)
            val width = (paint.measureText(fit(label, paint, 140f)) + 18f).coerceAtMost(150f)
            Triple(label, level, width)
        }
        val used = sized.sumOf { it.third.toDouble() }.toFloat() + 4f * (sized.size - 1).coerceAtLeast(0)
        var x = rect.right - 10f - used
        val pillTop = rect.top + 6f
        for ((label, level, width) in sized) {
            if (label.isEmpty() || x + width > rect.right - 8f) continue
            if (x < rect.left + 118f) break
            drawPill(canvas, x, pillTop, 14f, label, level, width)
            x += width + 4f
        }
        val captionTop = rect.bottom - 16f
        val well = RectF(rect.left + 8f, rect.top + 24f, rect.right - 8f, captionTop - 4f)
        canvas.drawRoundRect(well, 8f, 8f, fill(photoWell))
        val path = photo["path"] as? String
        var bitmap: Bitmap? = null
        try {
            if (!path.isNullOrBlank()) bitmap = decode(path)
            if (bitmap != null) {
                drawContained(canvas, bitmap, RectF(well.left + 6f, well.top + 6f, well.right - 6f, well.bottom - 6f))
            } else {
                val message = if (path.isNullOrBlank()) "No image" else "Image unavailable"
                val paint = textPaint(10f, Color.parseColor("#CBD5E1"), false)
                canvas.drawText(message, well.centerX() - paint.measureText(message) / 2f, well.centerY(), paint)
            }
        } finally {
            bitmap?.recycle()
        }
        val caption = textPaint(7.5f, metaInk, false)
        val strong = textPaint(7.5f, ink, true)
        canvas.drawText("Image Quality: ", rect.left + 10f, captionTop + 11f, strong)
        canvas.drawText(
            fit(photo["quality"] as? String ?: "Not assessed", caption, 120f),
            rect.left + 10f + strong.measureText("Image Quality: "),
            captionTop + 11f,
            caption,
        )
        val summaryLabel = "Summary: "
        val summaryText = fit(photo["summary"] as? String ?: "", caption, rect.width() * 0.48f)
        val summaryX = rect.right - 10f - caption.measureText(summaryText) - strong.measureText(summaryLabel)
        canvas.drawText(summaryLabel, summaryX, captionTop + 11f, strong)
        canvas.drawText(summaryText, summaryX + strong.measureText(summaryLabel), captionTop + 11f, caption)
    }

    private fun drawFooter(canvas: Canvas, text: String, page: String, baseline: Float) {
        canvas.drawRect(MARGIN, baseline - 12f, PAGE_W - MARGIN, baseline - 11f, fill(lineColor))
        val paint = textPaint(7f, labelInk, false)
        canvas.drawText(fit(text, paint, PAGE_W - MARGIN * 2 - 70f), MARGIN, baseline, paint)
        val pagePaint = textPaint(7f, labelInk, true)
        canvas.drawText(page, PAGE_W - MARGIN - pagePaint.measureText(page), baseline, pagePaint)
    }

    private fun drawPill(
        canvas: Canvas,
        x: Float,
        y: Float,
        height: Float,
        label: String,
        level: String,
        maxWidth: Float,
    ): Float {
        val tone = tone(level)
        val paint = textPaint(7f, tone.ink, true)
        val shown = fit(label, paint, maxWidth - 18f)
        val width = (paint.measureText(shown) + 18f).coerceAtMost(maxWidth)
        val rect = RectF(x, y, x + width, y + height)
        canvas.drawRoundRect(rect, height / 2f, height / 2f, fill(tone.tint))
        canvas.drawCircle(x + 8f, y + height / 2f, 2.8f, fill(tone.solid))
        canvas.drawText(shown, x + 13f, centerText(paint, y, height), paint)
        return width
    }

    private fun drawContained(canvas: Canvas, bitmap: Bitmap, dest: RectF) {
        val scale = minOf(dest.width() / bitmap.width, dest.height() / bitmap.height)
        if (scale <= 0f) return
        val width = bitmap.width * scale
        val height = bitmap.height * scale
        canvas.drawBitmap(
            bitmap, null,
            RectF(
                dest.centerX() - width / 2f,
                dest.centerY() - height / 2f,
                dest.centerX() + width / 2f,
                dest.centerY() + height / 2f,
            ),
            Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG),
        )
    }

    private fun decode(path: String): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1600) sample *= 2
        return BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
    }

    private fun card(canvas: Canvas, rect: RectF, color: Int) {
        canvas.drawRoundRect(rect, 8f, 8f, fill(color))
        canvas.drawRoundRect(rect, 8f, 8f, stroke(lineColor, 1f))
    }

    private fun fill(color: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color }

    private fun stroke(color: Int, width: Float) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        this.color = color
        style = Paint.Style.STROKE
        strokeWidth = width
    }

    private fun textPaint(size: Float, color: Int, bold: Boolean) =
        TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = size
            this.color = color
            typeface = if (bold) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }

    private fun centerText(paint: Paint, top: Float, height: Float) =
        top + height / 2f - (paint.ascent() + paint.descent()) / 2f

    private fun staticLayout(text: String, paint: TextPaint, width: Int, lines: Int): StaticLayout =
        StaticLayout.Builder.obtain(text, 0, text.length, paint, width.coerceAtLeast(1))
            .setAlignment(Layout.Alignment.ALIGN_NORMAL)
            .setMaxLines(lines)
            .setEllipsize(TextUtils.TruncateAt.END)
            .build()

    private fun fit(text: String, paint: Paint, maxWidth: Float): String {
        if (maxWidth <= 0f) return ""
        if (paint.measureText(text) <= maxWidth) return text
        val ellipsis = "…"
        var end = text.length
        while (end > 0 && paint.measureText(text.substring(0, end) + ellipsis) > maxWidth) end--
        return if (end == 0) ellipsis else text.substring(0, end) + ellipsis
    }

    private fun shown(value: String?) = if (value.isNullOrBlank()) "—" else value

    private fun mapList(value: Any?): List<Map<String, Any?>> {
        val raw = value as? List<*> ?: return emptyList()
        return raw.mapNotNull { item ->
            val map = item as? Map<*, *> ?: return@mapNotNull null
            buildMap {
                for ((key, entry) in map) {
                    if (key != null) put(key.toString(), entry)
                }
            }
        }
    }
}
